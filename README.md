The Aetherial Lattice
=====================

**The Aetherial Lattice** is a high-performance, audio-reactive generative art engine written in Python. It combines real-time signal processing (FFT) with a vectorized physics simulation to visualize sound as a living, breathing gravitational field.

🌟 Features
-----------

-   **Vectorized Physics Engine:** Utilizing `NumPy` for matrix operations, the engine calculates gravitational forces, friction, and collision avoidance for 1,500+ particles simultaneously, offering significantly higher performance than standard Python loops.

-   **Real-Time Audio Analysis:** Uses Fast Fourier Transform (FFT) to deconstruct live microphone input into Bass, Mid, and Treble energy bands.

    -   **Bass:** Controls gravitational force (repulsion/expansion).

    -   **Mids:** Controls atmospheric viscosity and color brightness.

    -   **Treble:** Controls entropy (particle jitter) and visual decay trails.

-   **Graceful Degradation (Autopilot):** If no microphone or audio driver is detected, the system automatically switches to a "Synthetic Mode," generating procedural waveforms so the visualization never crashes.

-   **Interactive Gravity:** The user's mouse acts as a gravitational anomaly (Singularity), allowing for direct interaction with the particle field.

🛠️ Installation
----------------

Ensure you have Python 3.8 or newer installed.

1.  **Clone the repository** (or download the script):

    ```
    git clone https://github.com/mh-bagheri/aetherial-lattice.git
    cd aetherial-lattice
    ```

2.  Install Dependencies:

    The project relies on pygame for rendering and numpy for math. pyaudio and scipy are optional but recommended for live audio reactivity.

    ```
    pip install pygame numpy scipy pyaudio
    ```

    *Note: If you have trouble installing `pyaudio` (common on some systems), the simulation will still run perfectly in Synthetic Mode.*

🚀 Usage
--------

Run the main script:

```
python aetherial_lattice.py
```

### Controls

-   **Mouse Move:** Influence the "wind" and direction of the lattice.

-   **Mouse Left Click + Hold:** Create a High-Gravity Singularity (Black Hole) that attracts particles and forms connection lines.

-   **ESC / Close Window:** Quit the application.

🧠 How It Works
---------------

The simulation is built on a hybrid architecture:

1.  **The Signal Processor:** Captures raw byte streams from the input device, applies a Hanning window, and performs an FFT to extract frequency magnitude.

2.  The Vector Engine:

    Instead of updating particles one by one:

    ```
    # Slow (Standard Python)
    for p in particles:
        p.x += p.vx
    ```

    We use Linear Algebra to update the entire state space in one CPU cycle:

    ```
    # Fast (Aetherial Lattice)
    self.pos += self.vel * dt
    ```

📄 License
----------

This project is open source and available under the MIT License.
