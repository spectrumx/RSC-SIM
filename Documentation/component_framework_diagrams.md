# RSC-SIM Component Framework Diagrams

This document contains all the diagrams from the RSC-SIM Component Framework documentation.

---

## 🧩 Component Overview

```mermaid
graph TB
    %% External Systems
    subgraph External["🌐 External Systems"]
        Files[Data Files<br/>.cut, .arrow, .tif]
        User[User Interface<br/>Tutorials & Scripts]
        Output[Output Systems<br/>Plots, Files, Reports]
    end

    %% Core Framework
    subgraph Framework["🏗️ RSC-SIM Framework"]
        %% Data Layer
        subgraph DataLayer["📊 Data Management Layer"]
            DataLoader[Data Loader<br/>radio_io.py]
            TypeSystem[Type System<br/>radio_types.py]
            Constants[Constants<br/>RadioMdl.py]
        end

        %% Processing Layer
        subgraph ProcessingLayer["⚙️ Processing Layer"]
            CoordEngine[Coordinate Engine<br/>coord_frames.py]
            AntennaEngine[Antenna Engine<br/>antenna_pattern.py]
            AstroEngine[Astronomy Engine<br/>astro_mdl.py]
            SatEngine[Satellite Engine<br/>sat_mdl.py]
            EnvEngine[Environment Engine<br/>env_mdl.py]
        end

        %% Integration Layer
        subgraph IntegrationLayer["🔗 Integration Layer"]
            ObsEngine[Observation Engine<br/>obs_mdl.py]
            LinkBudget[Link Budget Calculator]
            EnvIntegrator[Environmental Integrator]
        end

        %% Application Layer
        subgraph ApplicationLayer["📱 Application Layer"]
            TutorialManager[Tutorial Manager]
            SimulationEngine[Simulation Engine]
            AnalysisEngine[Analysis Engine]
        end
    end

    %% Data Flow
    Files --> DataLoader
    DataLoader --> TypeSystem
    TypeSystem --> Constants
    
    TypeSystem --> CoordEngine
    TypeSystem --> AntennaEngine
    TypeSystem --> AstroEngine
    TypeSystem --> SatEngine
    TypeSystem --> EnvEngine
    
    CoordEngine --> ObsEngine
    AntennaEngine --> ObsEngine
    AstroEngine --> ObsEngine
    SatEngine --> LinkBudget
    EnvEngine --> EnvIntegrator
    
    LinkBudget --> ObsEngine
    EnvIntegrator --> ObsEngine
    
    ObsEngine --> SimulationEngine
    SimulationEngine --> AnalysisEngine
    AnalysisEngine --> Output
    
    User --> TutorialManager
    TutorialManager --> SimulationEngine

    %% Styling
    classDef external fill:#E8F4F8,stroke:#333,stroke-width:2px
    classDef data fill:#F0F0F0,stroke:#333,stroke-width:2px
    classDef processing fill:#FFF4E6,stroke:#333,stroke-width:2px
    classDef integration fill:#E8F5E9,stroke:#333,stroke-width:2px
    classDef application fill:#F3E5F5,stroke:#333,stroke-width:2px

    class Files,User,Output external
    class DataLoader,TypeSystem,Constants data
    class CoordEngine,AntennaEngine,AstroEngine,SatEngine,EnvEngine processing
    class ObsEngine,LinkBudget,EnvIntegrator integration
    class TutorialManager,SimulationEngine,AnalysisEngine application
```

---

## 🔄 Component Interaction Flow

```mermaid
sequenceDiagram
    participant U as User
    participant TM as Tutorial Manager
    participant DL as Data Loader
    participant TS as Type System
    participant SE as Satellite Engine
    participant EE as Environment Engine
    participant LB as Link Budget
    participant OE as Observation Engine
    participant AE as Analysis Engine
    participant O as Output

    U->>TM: Start Tutorial
    TM->>DL: Load Data Files
    DL->>TS: Create Data Types
    TS->>SE: Initialize Satellite Models
    TS->>EE: Initialize Environment Models
    
    TM->>LB: Configure Link Budget
    LB->>SE: Get Satellite Parameters
    LB->>EE: Get Environmental Factors
    
    TM->>OE: Run Observation
    OE->>SE: Process Satellites
    OE->>EE: Apply Environmental Effects
    OE->>LB: Calculate Link Budget
    OE->>AE: Return Results
    
    AE->>O: Generate Plots/Reports
    O->>U: Display Results
```

---

## 🔌 Component Interfaces

### Data Flow Interfaces

```mermaid
graph LR
    subgraph Input["📥 Input Interfaces"]
        A1[Antenna Pattern Loader]
        A2[Trajectory Loader]
        A3[DEM Data Loader]
    end

    subgraph Processing["⚙️ Processing Interfaces"]
        B1[Coordinate Transform]
        B2[Gain Calculation]
        B3[Link Budget]
        B4[Environmental Effects]
    end

    subgraph Output["📤 Output Interfaces"]
        C1[Temperature Results]
        C2[Power Analysis]
        C3[Visualization]
        C4[Reports]
    end

    A1 --> B2
    A2 --> B1
    A3 --> B4
    B1 --> B3
    B2 --> B3
    B4 --> B3
    B3 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
```

---

## 🔧 Component Configuration

### Component Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Initialized: Component Creation
    Initialized --> Configured: Load Configuration
    Configured --> Ready: Initialize Dependencies
    Ready --> Processing: Start Simulation
    Processing --> Processing: Process Data
    Processing --> Completed: Simulation Complete
    Completed --> Ready: Reset for Next Run
    Ready --> [*]: Component Cleanup
```

---

## 🔄 Component Evolution

### Component Lifecycle Management

```mermaid
graph LR
    A[Component Creation] --> B[Testing & Validation]
    B --> C[Integration]
    C --> D[Deployment]
    D --> E[Monitoring]
    E --> F[Maintenance]
    F --> G[Retirement]
    G --> A
```

---

**Last Updated:** 2025  
**Framework Version:** 1.2.0  
**Component Count:** 8 Core + 7 Tutorial + 5 Application
