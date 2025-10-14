# RSC-SIM Architecture Diagram (Mermaid)

This file contains a Mermaid diagram that can be rendered in GitHub, GitLab, and many documentation tools.

## Full Architecture Diagram

```mermaid
graph TB
    %% Styling
    classDef coreType fill:#E8F4F8,stroke:#333,stroke-width:2px
    classDef modeling fill:#FFF4E6,stroke:#333,stroke-width:2px
    classDef io fill:#F0F0F0,stroke:#333,stroke-width:2px
    classDef utility fill:#E8F5E9,stroke:#333,stroke-width:2px
    classDef env fill:#F3E5F5,stroke:#333,stroke-width:2px
    classDef app fill:#FFFACD,stroke:#333,stroke-width:2px

    %% Application Layer
    subgraph Application["📱 Application Layer"]
        EDU["Educational Tutorials<br/>01-07<br/>Progressive Learning"]:::app
        RES["Research Tutorials<br/>Production Examples<br/>Real Data"]:::app
    end

    %% Observation Modeling Layer
    subgraph ObsLayer["🔭 Observation Modeling Layer"]
        OBS["obs_mdl.py<br/>───────────<br/>model_observed_temp()<br/>model_observed_temp_vectorized()<br/>+ atmospheric_refraction()"]:::modeling
    end

    %% Modeling Modules
    subgraph ModLayer["🧮 Modeling Modules"]
        ASTRO["astro_mdl.py<br/>───────────<br/>• estim_casA_flux()<br/>• power ↔ temp<br/>• antenna_mdl_ITU()"]:::modeling
        SAT["sat_mdl.py<br/>───────────<br/>• sat_link_budget()<br/>• Doppler effects<br/>• Transmitter chars<br/>• Environmental"]:::modeling
        ENV["env_mdl.py<br/>───────────<br/>• Terrain masking<br/>• Atmospheric refraction<br/>• Water vapor<br/>• DEM analysis"]:::env
    end

    %% Utility Layer
    subgraph UtilLayer["🛠️ Utility Functions"]
        COORD["coord_frames.py<br/>───────────<br/>• ground ↔ beam<br/>• coordinate transforms<br/>• radial velocity"]:::utility
        ANT["antenna_pattern.py<br/>───────────<br/>• interpolate_gain()<br/>• gain ↔ aperture<br/>• pattern mapping"]:::utility
    end

    %% Core Layer
    subgraph CoreLayer["⚙️ Core Types & I/O"]
        RADIO["RadioMdl.py<br/>───────────<br/>k_boltz, speed_c, rad"]:::coreType
        TYPES["radio_types.py<br/>───────────<br/>Antenna<br/>Instrument<br/>Trajectory<br/>Observation<br/>Constellation<br/>Transmitter"]:::coreType
        IO["radio_io.py<br/>───────────<br/>.cut files<br/>.arrow files<br/>.csv files"]:::io
    end

    %% Data Flow Connections
    EDU --> OBS
    RES --> OBS
    
    OBS --> ASTRO
    OBS --> SAT
    OBS --> ENV
    
    ASTRO --> ANT
    ASTRO --> TYPES
    
    SAT --> COORD
    SAT --> ANT
    SAT --> TYPES
    
    ENV --> TYPES
    ENV --> IO
    
    COORD --> RADIO
    ANT --> RADIO
    
    TYPES --> RADIO
    TYPES --> IO
    TYPES --> ANT
```

## Simplified Data Flow

```mermaid
flowchart LR
    A["Load Data<br/>.cut .arrow .tif"] --> B["Create Types<br/>Antenna to Instrument"]
    B --> C["Define Models<br/>Sky + Link Budget"]
    C --> D["Run Simulation<br/>model_observed_temp()"]
    D --> E["Analyze Results<br/>Plots + PSD"]
    
    style A fill:#E8F4F8
    style B fill:#FFF4E6
    style C fill:#E8F5E9
    style D fill:#F3E5F5
    style E fill:#FFFACD
```

## Module Dependencies

```mermaid
graph LR
    %% Core
    RadioMdl[RadioMdl.py<br/>Constants]
    radio_io[radio_io.py<br/>File I/O]
    radio_types[radio_types.py<br/>Data Types]
    
    %% Utility
    coord_frames[coord_frames.py<br/>Coordinates]
    antenna_pattern[antenna_pattern.py<br/>Patterns]
    
    %% Modeling
    astro_mdl[astro_mdl.py<br/>Astronomy]
    sat_mdl[sat_mdl.py<br/>Satellites]
    env_mdl[env_mdl.py<br/>Environment]
    obs_mdl[obs_mdl.py<br/>Observations]
    
    %% Dependencies
    radio_types --> RadioMdl
    radio_types --> radio_io
    radio_types --> antenna_pattern
    
    coord_frames --> RadioMdl
    antenna_pattern --> RadioMdl
    
    astro_mdl --> RadioMdl
    astro_mdl --> antenna_pattern
    
    sat_mdl --> radio_types
    sat_mdl --> coord_frames
    sat_mdl --> antenna_pattern
    
    env_mdl --> radio_io
    
    obs_mdl --> radio_types
    obs_mdl --> coord_frames
    obs_mdl --> sat_mdl
    obs_mdl --> env_mdl
    
    style RadioMdl fill:#E8F4F8
    style radio_io fill:#F0F0F0
    style radio_types fill:#E8F4F8
    style coord_frames fill:#E8F5E9
    style antenna_pattern fill:#E8F5E9
    style astro_mdl fill:#FFF4E6
    style sat_mdl fill:#FFF4E6
    style env_mdl fill:#F3E5F5
    style obs_mdl fill:#FFF4E6
```

## Class Relationships

```mermaid
classDiagram
    class Antenna {
        +DataFrame gain_pat
        +RegularGridInterpolator gain_func
        +float rad_eff
        +Tuple valid_freqs
        +from_file()
        +from_dataframe()
        +get_gain_values()
    }
    
    class Instrument {
        +Antenna antenna
        +float phy_temp
        +float cent_freq
        +float bw
        +callable signal_func
        +int freq_chan
        +List coords
        +from_scalar()
        +get_center_freq_chans()
    }
    
    class Trajectory {
        +DataFrame traj
        +from_file()
        +get_traj_between()
        +get_time_stamps()
    }
    
    class Observation {
        +Trajectory pts
        +Instrument inst
        +ndarray result
        +from_dates()
    }
    
    class Constellation {
        +DataFrame sats
        +Instrument tmt
        +callable lnk_bdgt_mdl
        +from_observation()
        +from_file()
    }
    
    class Transmitter {
        +Instrument instrument
        +str polarization
        +float polarization_angle
        +List harmonics
        +add_harmonic()
    }
    
    Instrument *-- Antenna : contains
    Observation *-- Trajectory : contains
    Observation *-- Instrument : contains
    Constellation *-- Instrument : contains
    Transmitter *-- Instrument : contains
```

## Physics Pipeline

```mermaid
flowchart TD
    Start[Start Simulation] --> LoadData[Load Data]
    
    LoadData --> CreateTypes[Create Types]
    CreateTypes --> DefineModels[Define Models]
    
    DefineModels --> SkyModel[Sky Model<br/>T_sky = Cas A + Atmosphere + CMB]
    DefineModels --> LinkBudget[Link Budget Model<br/>FSPL + Doppler + Transmitter]
    DefineModels --> EnvEffects[Environmental Effects<br/>Terrain + Atmosphere + Water vapor]
    
    SkyModel --> Integrate[Integrate All Physics]
    LinkBudget --> Integrate
    EnvEffects --> Integrate
    
    Integrate --> Compute[Compute System Temperature<br/>T_sys = T_RX + T_A]
    
    Compute --> Output[Output<br/>Temperature Time Series]
    
    Output --> Analyze[Analysis]
    Analyze --> TempPower[Temp ↔ Power]
    Analyze --> PSD[Power Spectral Density]
    Analyze --> Plots[Time Series Plots]
    
    style Start fill:#E8F4F8
    style LoadData fill:#FFF4E6
    style CreateTypes fill:#E8F5E9
    style DefineModels fill:#F3E5F5
    style SkyModel fill:#FFFACD
    style LinkBudget fill:#FFFACD
    style EnvEffects fill:#FFFACD
    style Integrate fill:#FFE4E1
    style Compute fill:#FFE4E1
    style Output fill:#E0FFE0
    style Analyze fill:#E0FFE0
```

## Environmental Effects Flow

```mermaid
flowchart LR
    Input[Satellite Position<br/>Alt, Az, Range] --> TerrainCheck{Terrain<br/>Masking?}
    
    TerrainCheck -->|Blocked| Block[Factor = 0.0<br/>Not Visible]
    TerrainCheck -->|Visible| AtmCheck[Atmospheric<br/>Effects]
    
    AtmCheck --> Refraction[Refraction<br/>Correction]
    AtmCheck --> WaterVapor[Water Vapor<br/>Absorption]
    AtmCheck --> AtmLoss[Atmospheric<br/>Loss]
    
    Refraction --> Combine[Combine Effects]
    WaterVapor --> Combine
    AtmLoss --> Combine
    
    Combine --> Factor[Environmental<br/>Factor: 0.0-1.0]
    Block --> Factor
    
    Factor --> LinkBudget[Apply to<br/>Link Budget]
    
    style Input fill:#E8F4F8
    style TerrainCheck fill:#FFF4E6
    style Block fill:#FFB6C1
    style AtmCheck fill:#E8F5E9
    style Refraction fill:#F3E5F5
    style WaterVapor fill:#F3E5F5
    style AtmLoss fill:#F3E5F5
    style Combine fill:#FFFACD
    style Factor fill:#E0FFE0
    style LinkBudget fill:#E0FFE0
```

## Vectorization Strategy

```mermaid
flowchart TD
    Data[Input Data<br/>T times, S satellites, F frequencies] --> Strategy{Vectorization<br/>Strategy}
    
    Strategy --> Loop[Loop over Time<br/>T iterations]
    
    Loop --> VecSat[Vectorize Satellites<br/>Process S satellites at once]
    
    VecSat --> VecFreq[Vectorize Frequencies<br/>Process F frequencies at once]
    
    VecFreq --> VecPoint[Vectorize Pointings<br/>Process P pointings at once]
    
    VecPoint --> Result[Result Array<br/>Shape: T × P × F]
    
    style Data fill:#E8F4F8
    style Strategy fill:#FFF4E6
    style Loop fill:#E8F5E9
    style VecSat fill:#F3E5F5
    style VecFreq fill:#FFFACD
    style VecPoint fill:#FFE4E1
    style Result fill:#E0FFE0
```

---

## How to View These Diagrams

### GitHub/GitLab
These Mermaid diagrams will render automatically when viewing this file on GitHub or GitLab.

### VS Code
Install the "Markdown Preview Mermaid Support" extension to see the diagrams in VS Code.

### Online
Copy the Mermaid code and paste it into https://mermaid.live/ for interactive viewing.

### Export
Use the Mermaid CLI or online tools to export as PNG, SVG, or PDF:
```bash
# Install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Generate PNG
mmdc -i ARCHITECTURE_MERMAID.md -o architecture_diagram.png
```

---

## Legend

**Colors:**
- 🔵 Light Blue - Core Types & Constants
- 🟡 Light Yellow - Application Layer
- 🟢 Light Green - Utility Functions
- 🟣 Light Purple - Environmental Effects
- 🟠 Light Orange - Modeling Modules
- ⚪ Light Gray - I/O Operations

