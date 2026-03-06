"""Adaptive Global Bundle Adjustment (AGBA) Backend.

To mitigate long-term drift in large-scale and complex environments,
we introduce an adaptive global bundle adjustment (AGBA) strategy.

Key features:
1. Adaptive trigger: global BA is triggered when local optimization
   residuals exceed a predefined threshold.
2. Chebyshev distance matrix for keyframe spatial relationships.
3. Two-hop suppression rule for graph sparsification.
4. Base edges between temporally adjacent keyframes.
5. Candidate edges sorted by Chebyshev distance, added iteratively.

Reference: Section 3.3 "AGBA module" in the paper.
"""

import torch
import lietorch
import numpy as np
from collections import defaultdict

from lietorch import SE3
from .factor_graph import FactorGraph


class DroidBackend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.upsample = args.upsample
        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms

        # AGBA parameters
        self.agba_residual_thresh = getattr(args, 'agba_residual_thresh', 0.5)
        self.agba_max_edges = getattr(args, 'agba_max_edges', 5000)
        self.agba_chebyshev_thresh = getattr(args, 'agba_chebyshev_thresh', 50.0)

        # Track local optimization residuals for adaptive triggering
        self.last_residual = 0.0

    def should_trigger_global_ba(self, residual):
        """Determine whether to trigger global BA based on residual threshold.

        When the local optimization residual exceeds the predefined threshold
        — typically indicating a substantial motion change — the system
        performs a full BA over all historical keyframes.

        Args:
            residual: current local optimization residual
        Returns:
            True if global BA should be triggered
        """
        return residual > self.agba_residual_thresh

    def compute_chebyshev_distance(self, positions):
        """Compute Chebyshev distance matrix between keyframe positions.

        D_ij = max(|x_i - x_j|, |y_i - y_j|, |z_i - z_j|)

        This metric captures the maximal axis-wise spatial difference and
        is effective for identifying meaningful structural relationships
        while preserving sparsity.

        Args:
            positions: keyframe positions (N, 3) tensor
        Returns:
            Distance matrix D (N, N) tensor
        """
        N = positions.shape[0]
        # Expand for pairwise computation
        pos_i = positions.unsqueeze(1).expand(N, N, 3)
        pos_j = positions.unsqueeze(0).expand(N, N, 3)

        # Chebyshev distance: max of absolute differences along each axis
        D = torch.abs(pos_i - pos_j).max(dim=-1)[0]
        return D

    def build_agba_edges(self, positions, max_edges=None):
        """Build sparse pose graph edges using AGBA strategy.

        1. Establish base edges between temporally adjacent keyframes:
           E_base = {(i, i+1) | 0 <= i < N-1}
        2. Sort candidate edges by Chebyshev distance (ascending)
        3. Add edges iteratively with two-hop suppression

        Args:
            positions: keyframe positions (N, 3) tensor
            max_edges: maximum number of edges
        Returns:
            ii, jj: edge index tensors
        """
        if max_edges is None:
            max_edges = self.agba_max_edges

        N = positions.shape[0]
        if N < 2:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        # Step 1: Base edges (temporally adjacent keyframes)
        edges = set()
        for i in range(N - 1):
            edges.add((i, i + 1))
            edges.add((i + 1, i))

        # Step 2: Compute Chebyshev distance matrix
        D = self.compute_chebyshev_distance(positions)

        # Build adjacency for hop counting
        adjacency = defaultdict(set)
        for (i, j) in edges:
            adjacency[i].add(j)

        # Step 3: Collect and sort candidate edges by distance
        candidates = []
        for i in range(N):
            for j in range(i + 2, N):  # Skip adjacent (already added)
                if D[i, j].item() < self.agba_chebyshev_thresh:
                    candidates.append((D[i, j].item(), i, j))

        candidates.sort(key=lambda x: x[0])

        # Step 4: Add edges with two-hop suppression
        suppressed = set()
        for dist, i, j in candidates:
            if len(edges) >= max_edges:
                break

            if (i, j) in suppressed:
                continue

            # Add bidirectional edge
            edges.add((i, j))
            edges.add((j, i))
            adjacency[i].add(j)
            adjacency[j].add(i)

            # Two-hop suppression: suppress edges within 2 hops of i or j
            self._apply_two_hop_suppression(i, j, adjacency, suppressed, N)

        # Convert to tensors
        if len(edges) == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        edge_list = list(edges)
        ii = torch.tensor([e[0] for e in edge_list], dtype=torch.long)
        jj = torch.tensor([e[1] for e in edge_list], dtype=torch.long)

        return ii, jj

    def _apply_two_hop_suppression(self, i, j, adjacency, suppressed, N):
        """Apply two-hop suppression rule.

        Upon adding a candidate edge (i, j), any other edge (k, l)
        within two graph hops of either i or j is suppressed:
            if min(hop(i,k), hop(j,l)) <= 2, then (k, l) is discarded.

        This avoids redundant connections while preserving global consistency.
        """
        # Get 1-hop and 2-hop neighbors of i and j
        neighbors_i = set()
        neighbors_i.add(i)
        for n1 in adjacency.get(i, set()):
            neighbors_i.add(n1)
            for n2 in adjacency.get(n1, set()):
                neighbors_i.add(n2)

        neighbors_j = set()
        neighbors_j.add(j)
        for n1 in adjacency.get(j, set()):
            neighbors_j.add(n1)
            for n2 in adjacency.get(n1, set()):
                neighbors_j.add(n2)

        # Suppress candidate edges where either endpoint is within 2 hops
        for k in neighbors_i:
            for l in neighbors_j:
                if k != i or l != j:
                    suppressed.add((k, l))
                    suppressed.add((l, k))

    @torch.no_grad()
    def __call__(self, steps=12):
        """ main update - Adaptive Global Bundle Adjustment """

        t = self.video.counter.value
        if not self.video.stereo and not torch.any(self.video.disps_sens):
            self.video.normalize()

        graph = FactorGraph(self.video, self.update_op, corr_impl="alt",
                           max_factors=16*t, upsample=self.upsample)

        # Extract keyframe positions for AGBA edge selection
        poses = SE3(self.video.poses[:t])
        positions = poses.translation()  # (N, 3) keyframe positions

        if positions.shape[0] > 2:
            # Build sparse pose graph using AGBA strategy
            ii, jj = self.build_agba_edges(positions.cpu())
            ii = ii.to(device=self.video.poses.device)
            jj = jj.to(device=self.video.poses.device)

            # Filter edges to valid range
            valid = (ii < t) & (jj < t) & (ii >= 0) & (jj >= 0)
            ii = ii[valid]
            jj = jj[valid]

            if len(ii) > 0:
                graph.add_factors(ii, jj)
        else:
            # Fallback for small number of keyframes
            graph.add_proximity_factors(rad=self.backend_radius,
                                        nms=self.backend_nms,
                                        thresh=self.backend_thresh,
                                        beta=self.beta)

        graph.update_lowmem(steps=steps)
        graph.clear_edges()
        self.video.dirty[:t] = True
