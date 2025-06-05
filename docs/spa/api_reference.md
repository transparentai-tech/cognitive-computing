# SPA API Reference

## Core Classes

### SemanticPointer

A semantic pointer representing a concept or symbol as a high-dimensional vector.

```python
class SemanticPointer:
    """
    A semantic pointer with HRR operations and vocabulary integration.
    
    Parameters
    ----------
    vector : np.ndarray
        The vector representation
    vocabulary : Vocabulary, optional
        Associated vocabulary for cleanup operations
    name : str, optional
        Name of this pointer in the vocabulary
    """
    
    def __init__(self, vector: np.ndarray, vocabulary: Optional[Vocabulary] = None, name: Optional[str] = None)
```

#### Methods

##### Mathematical Operations

```python
def bind(self, other: 'SemanticPointer') -> 'SemanticPointer':
    """Bind with another semantic pointer using circular convolution."""
    
def unbind(self, other: 'SemanticPointer') -> 'SemanticPointer':
    """Unbind another pointer using circular correlation."""
    
def __mul__(self, other: 'SemanticPointer') -> 'SemanticPointer':
    """Binding operator (circular convolution)."""
    
def __invert__(self) -> 'SemanticPointer':
    """Approximate inverse for unbinding."""
    
def __add__(self, other: 'SemanticPointer') -> 'SemanticPointer':
    """Bundle (superpose) with another pointer."""
    
def dot(self, other: 'SemanticPointer') -> float:
    """Compute dot product similarity."""
    
def __matmul__(self, other: Union['SemanticPointer', np.ndarray]) -> float:
    """Dot product operator."""
```

##### Properties

```python
@property
def vector(self) -> np.ndarray:
    """Get the underlying vector."""
    
@property
def dimension(self) -> int:
    """Get vector dimensionality."""
```

### Vocabulary

Collection of semantic pointers with parsing and cleanup capabilities.

```python
class Vocabulary:
    """
    Collection of semantic pointers with parsing and cleanup.
    
    Parameters
    ----------
    dimension : int
        Dimensionality of semantic pointers
    config : SPAConfig, optional
        Configuration for the vocabulary
    rng : np.random.RandomState, optional
        Random number generator for reproducibility
    """
    
    def __init__(self, dimension: int, config: Optional[SPAConfig] = None, rng: Optional[np.random.RandomState] = None)
```

#### Methods

```python
def create_pointer(self, name: str, vector: Optional[np.ndarray] = None) -> SemanticPointer:
    """Create or register a semantic pointer."""
    
def parse(self, expression: str) -> SemanticPointer:
    """Parse an expression into a semantic pointer."""
    
def cleanup(self, vector: np.ndarray, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
    """Find closest semantic pointers to a vector."""
    
def __getitem__(self, key: str) -> SemanticPointer:
    """Get pointer by name."""
    
def __contains__(self, key: str) -> bool:
    """Check if pointer exists."""
```

### SPA

Main SPA system coordinating modules and control.

```python
class SPA(CognitiveMemory):
    """
    Semantic Pointer Architecture system.
    
    Parameters
    ----------
    config : SPAConfig
        Configuration for the SPA system
    """
    
    def __init__(self, config: SPAConfig)
```

#### Methods

```python
def add_module(self, name: str, module: Module) -> None:
    """Add a cognitive module."""
    
def add_action(self, action: Action) -> None:
    """Add an action rule."""
    
def step(self, dt: Optional[float] = None) -> None:
    """Run one simulation step."""
    
def run(self, duration: float) -> None:
    """Run simulation for specified duration."""
```

### SPAConfig

Configuration for SPA systems.

```python
@dataclass
class SPAConfig(MemoryConfig):
    """Configuration for SPA models."""
    
    subdimensions: int = 16  # Dimensions per semantic pointer component
    neurons_per_dimension: int = 50  # For neural implementation
    max_similarity_matches: int = 10  # For cleanup
    threshold: float = 0.3  # Action selection threshold
    mutual_inhibition: float = 1.0  # Between actions
    bg_bias: float = 0.0  # Basal ganglia bias
    routing_inhibition: float = 3.0  # Thalamus inhibition
    synapse: float = 0.01  # Synaptic time constant
    dt: float = 0.001  # Simulation timestep
```

## Module Classes

### Module (Base Class)

Abstract base class for all SPA modules.

```python
class Module(ABC):
    """Base class for SPA modules."""
    
    def __init__(self, name: str, dimensions: int)
    
    @abstractmethod
    def update(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Update module state."""
```

### State

Represents and manipulates semantic pointer states with optional feedback.

```python
class State(Module):
    """
    State module with feedback.
    
    Parameters
    ----------
    name : str
        Module name
    dimensions : int
        Vector dimensionality
    feedback : float
        Feedback strength (0-1)
    """
    
    def __init__(self, name: str, dimensions: int, feedback: float = 0.0)
```

### Memory

Associative memory for semantic pointers using HRR.

```python
class Memory(Module):
    """
    Associative memory module.
    
    Parameters
    ----------
    name : str
        Module name
    dimensions : int
        Vector dimensionality
    capacity : int
        Maximum number of stored pairs
    """
    
    def __init__(self, name: str, dimensions: int, capacity: int = 100)
    
    def add_pair(self, key: SemanticPointer, value: SemanticPointer) -> None:
        """Store a key-value pair."""
    
    def recall(self, key: np.ndarray) -> Optional[np.ndarray]:
        """Recall value associated with key."""
```

### Buffer

Working memory buffer with gating control.

```python
class Buffer(Module):
    """
    Gated buffer module.
    
    Parameters
    ----------
    name : str
        Module name
    dimensions : int
        Vector dimensionality
    gate_default : float
        Default gate value (0-1)
    """
    
    def __init__(self, name: str, dimensions: int, gate_default: float = 1.0)
```

### Gate

Controls information flow between modules.

```python
class Gate(Module):
    """
    Gate module for controlling flow.
    
    Parameters
    ----------
    name : str
        Module name
    dimensions : int
        Vector dimensionality
    default_value : float
        Default gate value
    """
    
    def __init__(self, name: str, dimensions: int, default_value: float = 0.0)
```

### Compare

Computes similarity between semantic pointers.

```python
class Compare(Module):
    """
    Compare module for similarity.
    
    Parameters
    ----------
    name : str
        Module name
    dimensions : int
        Vector dimensionality
    """
    
    def __init__(self, name: str, dimensions: int)
```

## Action Selection Classes

### Action

Single action with condition and effect.

```python
class Action:
    """
    Action with condition and effect.
    
    Parameters
    ----------
    name : str
        Action name
    condition : Callable
        Function returning utility
    effect : Callable
        Function to execute
    """
    
    def __init__(self, name: str, condition: Callable[[], float], effect: Callable[[], None])
```

### BasalGanglia

Action selection through competition.

```python
class BasalGanglia:
    """
    Basal ganglia for action selection.
    
    Parameters
    ----------
    actions : List[Action]
        Available actions
    mutual_inhibition : float
        Inhibition between actions
    threshold : float
        Selection threshold
    """
    
    def __init__(self, actions: List[Action], mutual_inhibition: float = 1.0, threshold: float = 0.3)
    
    def update(self, state: Dict[str, Any]) -> np.ndarray:
        """Evaluate actions and compute outputs."""
```

### Thalamus

Routes selected actions to modules.

```python
class Thalamus:
    """
    Thalamus for action routing.
    
    Parameters
    ----------
    basal_ganglia : BasalGanglia
        Connected basal ganglia
    modules : Dict[str, Module]
        Modules to route to
    routing_inhibition : float
        Inhibition strength
    """
    
    def __init__(self, basal_ganglia: BasalGanglia, modules: Dict[str, Module], routing_inhibition: float = 3.0)
```

## Production System Classes

### Production

IF-THEN production rule.

```python
class Production:
    """
    Production rule.
    
    Parameters
    ----------
    name : str
        Production name
    condition : Condition
        Matching condition
    effect : Effect
        Effect to apply
    priority : float
        Priority for conflict resolution
    """
    
    def __init__(self, name: str, condition: 'Condition', effect: 'Effect', priority: float = 1.0)
```

### ProductionSystem

Collection of productions with conflict resolution.

```python
class ProductionSystem:
    """
    Production system with conflict resolution.
    
    Parameters
    ----------
    conflict_resolution : str
        Strategy ('priority', 'recency', 'specificity')
    """
    
    def __init__(self, conflict_resolution: str = "priority")
    
    def add_production(self, production: Production) -> None:
        """Add a production rule."""
    
    def match(self, state: Dict[str, Any]) -> List[Production]:
        """Find matching productions."""
```

## Control Classes

### CognitiveControl

Executive control over SPA modules.

```python
class CognitiveControl(Module):
    """
    Cognitive control for executive functions.
    
    Parameters
    ----------
    dimensions : int
        Vector dimensionality
    config : SPAConfig
        Configuration
    vocab : Vocabulary, optional
        Control vocabulary
    """
    
    def __init__(self, dimensions: int, config: SPAConfig, vocab: Optional[Vocabulary] = None)
    
    def set_attention(self, target: Union[str, np.ndarray]) -> None:
        """Set attention focus."""
    
    def set_goal(self, goal: Union[str, np.ndarray]) -> None:
        """Set current goal."""
```

### Sequencing

Sequential behavior control.

```python
class Sequencing(Module):
    """
    Sequential behavior control.
    
    Parameters
    ----------
    dimensions : int
        Vector dimensionality
    config : SPAConfig
        Configuration
    vocab : Vocabulary
        Sequence vocabulary
    """
    
    def __init__(self, dimensions: int, config: SPAConfig, vocab: Vocabulary)
    
    def define_sequence(self, name: str, steps: List[Union[str, Callable]]) -> None:
        """Define a sequence."""
    
    def start_sequence(self, name: str) -> None:
        """Start executing sequence."""
```

## Model Building Classes

### SPAModel

High-level SPA model specification.

```python
class SPAModel:
    """
    Declarative SPA model specification.
    
    Parameters
    ----------
    name : str
        Model name
    dimension : int
        Default dimensionality
    config : SPAConfig, optional
        Model configuration
    """
    
    def __init__(self, name: str, dimension: int = 512, config: Optional[SPAConfig] = None)
    
    def add_module(self, name: str, module_type: str, **kwargs) -> None:
        """Add a module specification."""
    
    def connect(self, source: str, target: str, transform: Optional[np.ndarray] = None) -> None:
        """Connect modules."""
```

### ModelBuilder

Builds executable SPA from specification.

```python
class ModelBuilder:
    """Build SPA system from specification."""
    
    def build(self, model: SPAModel) -> SPA:
        """Build complete SPA system."""
```

## Utility Functions

```python
def create_spa(dimension: int = 512, **kwargs) -> SPA:
    """Create SPA system with configuration."""

def create_vocabulary(dimension: int = 512, **kwargs) -> Vocabulary:
    """Create vocabulary with configuration."""

def make_unitary(pointer: np.ndarray) -> np.ndarray:
    """Make semantic pointer unitary for binding."""

def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""

def normalize_semantic_pointer(pointer: np.ndarray) -> np.ndarray:
    """Normalize to unit length."""

def generate_pointers(vocab_size: int, dimensions: int, rng: Optional[np.random.RandomState] = None) -> Dict[str, np.ndarray]:
    """Generate orthogonal semantic pointers."""

def analyze_vocabulary(vocab: Vocabulary) -> Dict[str, Any]:
    """Analyze vocabulary statistics."""

def parse_actions(action_spec: str) -> List[Action]:
    """Parse action rules from string."""
```

## Visualization Functions

```python
def plot_similarity_matrix(vocab: Vocabulary, **kwargs) -> Tuple[Figure, Axes]:
    """Plot vocabulary similarity matrix."""

def plot_action_selection(bg: BasalGanglia, history: np.ndarray, **kwargs) -> Tuple[Figure, Axes]:
    """Plot action selection dynamics."""

def plot_network_graph(network: Network, **kwargs) -> Tuple[Figure, Axes]:
    """Visualize network connectivity."""

def visualize_production_flow(production_system: ProductionSystem, **kwargs) -> Tuple[Figure, Axes]:
    """Visualize production system."""

def animate_state_evolution(states: List[np.ndarray], **kwargs) -> FuncAnimation:
    """Animate state changes over time."""
```

## Example Usage

### Basic Semantic Pointer Operations

```python
from cognitive_computing.spa import Vocabulary, SemanticPointer

# Create vocabulary
vocab = Vocabulary(512)
vocab.create_pointer("COFFEE")
vocab.create_pointer("HOT")

# Bind concepts
hot_coffee = vocab["HOT"] * vocab["COFFEE"]

# Unbind to retrieve component
coffee = hot_coffee * ~vocab["HOT"]

# Check similarity
similarity = coffee @ vocab["COFFEE"]  # Should be close to 1.0
```

### Building a Simple Model

```python
from cognitive_computing.spa import SPAModel, ModelBuilder

# Define model
model = SPAModel("simple_model", dimension=256)
model.add_module("working_memory", "state", feedback=0.9)
model.add_module("semantic_memory", "memory", capacity=100)
model.connect("working_memory", "semantic_memory")

# Build and run
builder = ModelBuilder()
spa = builder.build(model)
spa.run(duration=1.0)
```

### Action Selection Example

```python
from cognitive_computing.spa import Action, BasalGanglia, Thalamus

# Define actions
actions = [
    Action("eat", lambda: hunger_level, lambda: consume_food()),
    Action("drink", lambda: thirst_level, lambda: consume_water()),
    Action("sleep", lambda: fatigue_level, lambda: rest())
]

# Create selection system
bg = BasalGanglia(actions, threshold=0.5)
thal = Thalamus(bg, modules={'motor': motor_module})

# Update selection
bg.update(current_state)
thal.route_actions()
```

## Error Handling

All SPA classes include comprehensive error handling:

- **ValueError**: For invalid dimensions or parameters
- **KeyError**: For missing vocabulary items or modules
- **TypeError**: For incorrect types
- **RuntimeError**: For system state errors

Example:

```python
try:
    vocab = Vocabulary(dimension=10)  # Too small
except ValueError as e:
    print(f"Configuration error: {e}")

try:
    pointer = vocab["NONEXISTENT"]  # Key not found
except KeyError as e:
    print(f"Vocabulary error: {e}")
```