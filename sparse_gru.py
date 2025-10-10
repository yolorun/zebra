import torch
import torch.nn as nn
import torch.sparse


class SparseGRUBrain(nn.Module):
    """
    Efficiently trains 70k GRU models in parallel with sparse connectivity.
    Each neuron has its own GRU with potentially different input connections.
    """

    def __init__(self, num_neurons, hidden_dim, edge_list, stimulus_dim=0, bias=True):
        """
        Args:
            num_neurons: Number of neurons (70k in your case)
            hidden_dim: Hidden dimension for each GRU
            edge_list: List of (source, target) tuples defining ALL connections
                      including self-connections if they exist
            stimulus_dim: Dimension of stimulus input (0 if no stimulus)
            bias: Whether to use bias terms in GRU gates
        """
        super().__init__()
        N, H = num_neurons, hidden_dim
        self.num_neurons = N
        self.hidden_dim = H
        self.stimulus_dim = stimulus_dim

        # Build sparse indices from edge list
        self.register_buffer('W_indices', self._build_sparse_indices(edge_list, N, H))
        num_edges = len(edge_list)

        # Sparse input weights (calcium -> hidden) for each gate - keep in float32
        self.W_z_values = nn.Parameter(torch.randn(num_edges * H, dtype=torch.float32) * 0.01)
        self.W_r_values = nn.Parameter(torch.randn(num_edges * H, dtype=torch.float32) * 0.01)
        self.W_h_values = nn.Parameter(torch.randn(num_edges * H, dtype=torch.float32) * 0.01)

        # Dense batched recurrent weights (hidden -> hidden) for each neuron
        self.U_z = nn.Parameter(torch.randn(N, H, H) * 0.01)
        self.U_r = nn.Parameter(torch.randn(N, H, H) * 0.01)
        self.U_h = nn.Parameter(torch.randn(N, H, H) * 0.01)

        # Biases for each gate (per neuron)
        if bias:
            self.b_z = nn.Parameter(torch.zeros(N, H))
            self.b_r = nn.Parameter(torch.zeros(N, H))
            self.b_h = nn.Parameter(torch.zeros(N, H))
        else:
            self.register_parameter('b_z', None)
            self.register_parameter('b_r', None)
            self.register_parameter('b_h', None)

        # Stimulus projection layers (if stimulus is used)
        if stimulus_dim > 0:
            self.stimulus_projection_z = nn.Linear(stimulus_dim, N * H)
            self.stimulus_projection_r = nn.Linear(stimulus_dim, N * H)
            self.stimulus_projection_h = nn.Linear(stimulus_dim, N * H)
        
        # Output projection: hidden -> predicted calcium (scalar per neuron)
        # self.output_projection = nn.Parameter(torch.randn(N, H) * 0.01)
        self.output_projection = nn.Parameter(torch.randn(H))  # shared

        # Pre-build sparse tensor shapes for efficiency
        self.sparse_shape = (N * H, N)

        # self._build_csr_template()

    def _build_sparse_indices(self, edge_list, N, H):
        """
        Build sparse indices for efficient sparse matrix multiplication.
        Each edge (src, tgt) creates H connections in the sparse matrix.
        """
        indices = []
        for src, tgt in edge_list:
            # Each edge creates H connections (one per hidden dimension)
            for h in range(H):
                row = tgt * H + h  # target neuron's h-th hidden unit
                col = src  # source neuron's calcium value
                indices.append([row, col])

        return torch.LongTensor(indices).T  # Shape: (2, num_edges * H)

    # def _build_csr_template(self):
    #     """Build CSR template structure once to avoid repeated conversions."""
    #     # Create a dummy COO tensor with ones
    #     dummy_values = torch.ones(self.W_indices.shape[1])
    #     W_coo = torch.sparse_coo_tensor(
    #         self.W_indices,
    #         dummy_values,
    #         self.sparse_shape
    #     ).coalesce()
        
    #     # Convert to CSR and store the structure
    #     W_csr = W_coo.to_sparse_csr()
        
    #     # Store CSR indices (these don't change, only values do)
    #     self.register_buffer('csr_crow_indices', W_csr.crow_indices())
    #     self.register_buffer('csr_col_indices', W_csr.col_indices())

    # def _sparse_matmul(self, values, input_tensor):
    #     """
    #     Efficient sparse matrix multiplication with float32 weights.
    #     Forces float32 computation by disabling autocast for this operation.
    #     """
    #     # Store original dtype for casting back
    #     original_dtype = input_tensor.dtype
        
    #     # Disable autocast and force float32 for sparse operations
    #     with torch.cuda.amp.autocast(enabled=False):
    #         # Cast to float32
    #         input_fp32 = input_tensor.float()
    #         values_fp32 = values.float()
            
    #         # Create COO sparse tensor in float32
    #         W = torch.sparse_coo_tensor(
    #             self.W_indices,
    #             values_fp32,
    #             self.sparse_shape,
    #             dtype=torch.float32,
    #             device=input_tensor.device
    #         )
    #         W = W.coalesce()

    #         # Sparse matmul in float32
    #         output = torch.sparse.mm(W, input_fp32.T).T
        
    #     # Cast back to original dtype after leaving the no-autocast context
    #     if original_dtype != torch.float32:
    #         output = output.to(original_dtype)

    #     # Reshape to (B, N, H)
    #     B = input_tensor.shape[0]
    #     return output.reshape(B, self.num_neurons, self.hidden_dim)

    def _sparse_matmul(self, values, input_tensor):
        """
        Efficient sparse matrix multiplication with float32 weights.
        Forces float32 computation by disabling autocast for this operation.
        """
        # Store original dtype for casting back
        original_dtype = input_tensor.dtype
        
        # Disable autocast and force float32 for sparse operations
        with torch.cuda.amp.autocast(enabled=False):
            # Cast to float32
            input_fp32 = input_tensor.float()
            values_fp32 = values.float()
            
            # Create COO sparse tensor in float32
            W = torch.sparse_coo_tensor(
                self.W_indices,
                values_fp32,
                self.sparse_shape,
                dtype=torch.float32,
                device=input_tensor.device
            )
            W = W.coalesce()

            # Sparse matmul in float32
            output = torch.sparse.mm(W, input_fp32.T).T
        
        # Cast back to original dtype after leaving the no-autocast context
        if original_dtype != torch.float32:
            output = output.to(original_dtype)

        # Reshape to (B, N, H)
        B = input_tensor.shape[0]
        return output.reshape(B, self.num_neurons, self.hidden_dim)

    def forward(self, calcium_t, hidden, stimulus_t=None):
        """
        Forward pass through all 70k GRUs in parallel.

        Args:
            calcium_t: (B, N) - calcium values for all neurons at time t
            hidden: (B, N, H) - hidden states for all neurons
            stimulus_t: (B, stimulus_dim) - stimulus at time t (optional)

        Returns:
            calcium_t1: (B, N) - predicted calcium for all neurons at t+1
            hidden_new: (B, N, H) - updated hidden states
        """
        B, N = calcium_t.shape

        # Sparse input transforms via connectivity matrix
        inp_z = self._sparse_matmul(self.W_z_values, calcium_t)
        inp_r = self._sparse_matmul(self.W_r_values, calcium_t)
        inp_h = self._sparse_matmul(self.W_h_values, calcium_t)
        
        # Add stimulus contribution if provided
        if stimulus_t is not None and self.stimulus_dim > 0:
            stim_z = self.stimulus_projection_z(stimulus_t).view(B, N, self.hidden_dim)
            stim_r = self.stimulus_projection_r(stimulus_t).view(B, N, self.hidden_dim)
            stim_h = self.stimulus_projection_h(stimulus_t).view(B, N, self.hidden_dim)
            inp_z = inp_z + stim_z
            inp_r = inp_r + stim_r
            inp_h = inp_h + stim_h

        # Dense recurrent transforms (batched matrix multiply)
        # Each neuron has its own weight matrix U[n], so we use einsum for correct indexing
        # hidden: (B, N, H), U_z: (N, H, H) -> rec_z: (B, N, H)
        rec_z = torch.einsum('bnh,nhi->bni', hidden, self.U_z)
        rec_r = torch.einsum('bnh,nhi->bni', hidden, self.U_r)

        # GRU gate computations
        if self.b_z is not None:
            z = torch.sigmoid(inp_z + rec_z + self.b_z.unsqueeze(0))
            r = torch.sigmoid(inp_r + rec_r + self.b_r.unsqueeze(0))
        else:
            z = torch.sigmoid(inp_z + rec_z)
            r = torch.sigmoid(inp_r + rec_r)

        # Candidate hidden state
        # Apply reset gate then recurrent transform
        rec_h = torch.einsum('bnh,nhi->bni', r * hidden, self.U_h)

        if self.b_h is not None:
            h_tilde = torch.tanh(inp_h + rec_h + self.b_h.unsqueeze(0))
        else:
            h_tilde = torch.tanh(inp_h + rec_h)

        # Update hidden state
        hidden_new = (1 - z) * hidden + z * h_tilde

        # Output projection to predict next calcium value
        # calcium_t1 = (hidden_new * self.output_projection.unsqueeze(0)).sum(dim=-1)
        calcium_t1 = (hidden_new * self.output_projection).sum(dim=-1)
        calcium_t1 = torch.relu(calcium_t1)  # Ensure non-negative calcium values

        return calcium_t1, hidden_new

    def init_hidden(self, batch_size, device=None):
        """
        Initialize hidden states for all neurons.
        """
        if device is None:
            device = self.output_projection.device
        return torch.zeros(batch_size, self.num_neurons, self.hidden_dim, device=device)
    
    @classmethod
    def from_connectivity_graph(cls, connectivity_graph, hidden_dim, stimulus_dim=0, 
                                include_self_connections=True, min_strength=0.0, bias=True,
                                num_neurons=None, assume_zero_indexed=False):
        """
        Create SparseGRUBrain from connectivity graph dict.
        
        Args:
            connectivity_graph: Dict mapping post_synaptic -> [(pre_synaptic, strength, ...), ...]
            hidden_dim: Hidden dimension for each GRU
            stimulus_dim: Dimension of stimulus input
            include_self_connections: Whether to add self-connections
            min_strength: Minimum connection strength to include
            bias: Whether to use bias terms
            num_neurons: Total number of neurons (if None, inferred from graph)
            assume_zero_indexed: If True, assumes neuron IDs are 0-indexed, otherwise 1-indexed
        
        Returns:
            SparseGRUBrain instance
        """
        # Handle both 0-indexed and 1-indexed neuron IDs
        offset = 0 if assume_zero_indexed else 1
        
        if num_neurons is None:
            # Get all unique neurons from connectivity graph
            all_neurons = set()
            for post_syn in connectivity_graph.keys():
                all_neurons.add(post_syn)
                for connection in connectivity_graph[post_syn]:
                    all_neurons.add(connection[0])  # pre_synaptic
            
            # Convert to sorted list for consistent indexing
            neuron_list = sorted(list(all_neurons))
            neuron_to_idx = {neuron: neuron - offset for neuron in neuron_list}
            num_neurons = len(neuron_list)
        else:
            # Use provided number of neurons (0 to num_neurons-1)
            neuron_to_idx = {i + offset: i for i in range(num_neurons)}
        
        # Build edge list
        edge_list = set()
        edges_added = 0
        edges_skipped = 0
        
        for post_syn, connections in connectivity_graph.items():
            if post_syn not in neuron_to_idx:
                continue
            target_idx = neuron_to_idx[post_syn]
            
            for connection in connections:
                pre_syn = connection[0]
                strength = connection[1] if len(connection) > 1 else 1.0
                
                # Skip weak connections
                if strength < min_strength:
                    edges_skipped += 1
                    continue
                    
                if pre_syn in neuron_to_idx:
                    source_idx = neuron_to_idx[pre_syn]
                    edge_list.add((source_idx, target_idx))
                    edges_added += 1
        
        print(f"Added {edges_added} edges from connectivity graph (skipped {edges_skipped} weak connections)")
        
        # Add self-connections if requested
        if include_self_connections:
            for idx in range(num_neurons):
                edge_list.add((idx, idx))
        
        print(f"Creating SparseGRUBrain with {num_neurons} neurons and {len(edge_list)} connections")
        return cls(num_neurons, hidden_dim, list(edge_list), stimulus_dim=stimulus_dim, bias=bias)


# Example usage
def create_brain_model(num_neurons=70000, hidden_dim=32):
    """
    Example of creating the model with connectivity.
    """
    # Example connectivity: each neuron sees 10-50 random other neurons
    import random
    edge_list = []

    for target in range(num_neurons):
        # Random number of input connections per neuron
        num_inputs = random.randint(10, 50)
        sources = random.sample(range(num_neurons), num_inputs)

        for source in sources:
            edge_list.append((source, target))

        # Optional: add self-connection
        if random.random() > 0.5:
            edge_list.append((target, target))

    model = SparseGRUBrain(num_neurons, hidden_dim, edge_list)
    return model


# Training example
def train_step(model, calcium_sequence, optimizer):
    """
    Example training step with teacher forcing.

    Args:
        model: SparseGRUBrain instance
        calcium_sequence: (T, B, N) - sequence of calcium values
        optimizer: torch optimizer
    """
    T, B, N = calcium_sequence.shape
    hidden = model.init_hidden(B)

    loss = 0
    criterion = nn.MSELoss()

    for t in range(T - 1):
        # Predict next timestep
        calcium_pred, hidden = model(calcium_sequence[t], hidden)

        # Teacher forcing: use actual next value as input
        loss += criterion(calcium_pred, calcium_sequence[t + 1])

    # Backprop and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item() / (T - 1)
