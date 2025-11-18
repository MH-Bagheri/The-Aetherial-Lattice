"""
Project: The Aetherial Lattice
License: MIT
Description: A high-performance, audio-reactive particle physics engine.
             It visualizes sound as a living, breathing gravitational field.

Features:
- Real-time Audio Analysis (Fast Fourier Transform) via PyAudio & Scipy.
- Vectorized Physics Engine using NumPy (Gravity, Friction, repulsion).
- Graceful Degradation: Falls back to 'Synthetic Mode' if no microphone is found.
- Mouse Interaction: The cursor becomes a gravitational anomaly.

Dependencies:
    pip install pygame numpy scipy pyaudio
"""

import sys
import time
import math
import random
import dataclasses
from typing import Tuple, Optional, List

# --- Dependency Check & Imports ---
try:
    import pygame
    import numpy as np
except ImportError as e:
    print(f"CRITICAL: Missing core dependency. {e}")
    print("Please run: pip install pygame numpy")
    sys.exit(1)

# Audio libraries are optional; we have a fallback generator
AUDIO_AVAILABLE = False
try:
    import pyaudio
    from scipy.fftpack import fft
    AUDIO_AVAILABLE = True
except ImportError:
    print("WARNING: Audio libraries (pyaudio, scipy) not found.")
    print("Switching to SYNTHETIC MODE (Visuals will react to generated noise).")

# --- Configuration ---
@dataclasses.dataclass
class Config:
    WIDTH: int = 1200
    HEIGHT: int = 800
    FPS: int = 60
    PARTICLE_COUNT: int = 1500
    # Physics
    FRICTION: float = 0.96
    BASE_SPEED: float = 0.5
    CONNECTION_DISTANCE: int = 80
    # Audio Settings
    CHUNK_SIZE: int = 1024
    RATE: int = 44100
    # Colors
    BG_COLOR: Tuple[int, int, int] = (10, 10, 15)
    Palette: List[Tuple[int, int, int]] = dataclasses.field(
        default_factory=lambda: [(0, 255, 255), (255, 0, 128), (128, 0, 255), (255, 255, 255)]
    )

CFG = Config()

# --- Math & Physics Engine ---

class PhysicsEngine:
    """
    Handles the raw number crunching using NumPy for vectorization.
    This is significantly faster than iterating through Python objects.
    """
    def __init__(self, count: int, width: int, height: int):
        self.count = count
        self.width = width
        self.height = height
        
        # Position Matrix: [x, y] for all particles
        self.pos = np.random.rand(count, 2) * [width, height]
        
        # Velocity Matrix: [vx, vy]
        self.vel = (np.random.rand(count, 2) - 0.5) * 2
        
        # Attribute Matrix: [size, decay_rate]
        self.attrs = np.random.rand(count, 2)
        self.attrs[:, 0] = self.attrs[:, 0] * 2 + 1  # Size 1-3
        
        # Colors (indices into palette)
        self.colors = np.random.randint(0, len(CFG.Palette), count)

    def update(self, audio_energy: dict, mouse_pos: Tuple[int, int], mouse_pressed: bool):
        """
        Update particle physics based on audio inputs and mouse interaction.
        audio_energy keys: 'bass', 'mid', 'treble' (normalized 0.0 - 1.0)
        """
        bass = audio_energy.get('bass', 0)
        mid = audio_energy.get('mid', 0)
        treble = audio_energy.get('treble', 0)

        # 1. Apply base movement + Audio Excitement
        # Bass adds explosive force, Treble adds jittery vibration
        jitter = (np.random.rand(self.count, 2) - 0.5) * (treble * 2.0)
        self.vel += jitter
        
        # 2. Center Gravity / Repulsion (The "Heartbeat")
        # If bass is high, push away from center. If low, pull towards center.
        center = np.array([self.width / 2, self.height / 2])
        dir_to_center = center - self.pos
        dist_to_center = np.linalg.norm(dir_to_center, axis=1, keepdims=True)
        dist_to_center = np.maximum(dist_to_center, 1.0) # Avoid divide by zero
        
        norm_dir = dir_to_center / dist_to_center
        
        # Dynamic gravity: Negative means push away (Bass kick)
        gravity_strength = 0.05 - (bass * 0.25) 
        self.vel += norm_dir * gravity_strength

        # 3. Mouse Interaction (Black Hole / White Hole)
        if mouse_pressed:
            m_pos = np.array(mouse_pos)
            dir_to_mouse = m_pos - self.pos
            dist_mouse = np.linalg.norm(dir_to_mouse, axis=1, keepdims=True)
            dist_mouse = np.maximum(dist_mouse, 10.0)
            
            # Attract to mouse strongly
            self.vel += (dir_to_mouse / dist_mouse) * 1.5

        # 4. Apply Velocity & Friction
        self.pos += self.vel
        self.vel *= (CFG.FRICTION - (mid * 0.05)) # Mids make air "thicker"

        # 5. Screen Wrap
        self.pos[:, 0] = np.mod(self.pos[:, 0], self.width)
        self.pos[:, 1] = np.mod(self.pos[:, 1], self.height)

# --- Audio Handler ---

class AudioHandler:
    def __init__(self):
        self.stream = None
        self.p = None
        self.synthetic_t = 0.0
        
        if AUDIO_AVAILABLE:
            try:
                self.p = pyaudio.PyAudio()
                # Attempt to find default input device
                self.stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=CFG.RATE,
                    input=True,
                    frames_per_buffer=CFG.CHUNK_SIZE
                )
            except Exception as e:
                print(f"Audio init failed ({e}). Reverting to Synthetic Mode.")
                self.stream = None

    def get_energy(self) -> dict:
        """
        Returns normalized energy levels for Bass, Mid, Treble.
        Range: 0.0 to 1.0 (soft clamped)
        """
        if self.stream:
            try:
                # Read raw data
                data = np.frombuffer(self.stream.read(CFG.CHUNK_SIZE, exception_on_overflow=False), dtype=np.int16)
                
                # Perform FFT
                fft_data = np.abs(fft(data))[:CFG.CHUNK_SIZE // 2]
                
                # Normalize (logarithmic scaling helps visualization)
                fft_data = np.log10(fft_data + 1)
                
                # Define frequency bands (indices depend on RATE and CHUNK_SIZE)
                # Basic approx: Bass (0-100Hz), Mid (100-2000Hz), Treble (2k+)
                bass_range = fft_data[1:10] 
                mid_range = fft_data[10:100]
                treble_range = fft_data[100:]
                
                return {
                    'bass': np.clip(np.mean(bass_range) / 5, 0, 1),
                    'mid': np.clip(np.mean(mid_range) / 5, 0, 1),
                    'treble': np.clip(np.mean(treble_range) / 4, 0, 1)
                }
            except Exception:
                return self._generate_synthetic()
        else:
            return self._generate_synthetic()

    def _generate_synthetic(self) -> dict:
        """Generates procedural Perlin-like noise for demo mode."""
        self.synthetic_t += 0.05
        return {
            'bass': (math.sin(self.synthetic_t) + 1) / 2 * 0.8,
            'mid': (math.sin(self.synthetic_t * 1.5) + 1) / 2 * 0.5,
            'treble': (math.cos(self.synthetic_t * 3.0) + 1) / 2 * 0.3
        }

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

# --- Main Application ---

class AetherialLattice:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(f"Aetherial Lattice | Particles: {CFG.PARTICLE_COUNT}")
        self.screen = pygame.display.set_mode((CFG.WIDTH, CFG.HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.SysFont("monospace", 16)
        
        # Systems
        self.physics = PhysicsEngine(CFG.PARTICLE_COUNT, CFG.WIDTH, CFG.HEIGHT)
        self.audio = AudioHandler()
        
        # Visuals
        # We use a surface with alpha for the "trail" effect
        self.trail_surface = pygame.Surface((CFG.WIDTH, CFG.HEIGHT))
        self.trail_surface.set_alpha(60) # Lower = longer trails

    def draw_ui(self, energy):
        """Draws debug stats and audio bars."""
        fps = int(self.clock.get_fps())
        
        # Labels
        ui_text = [
            f"FPS: {fps}",
            f"Mode: {'LIVE MIC' if self.audio.stream else 'SYNTHETIC'}",
            "Controls: Mouse Click to Attract"
        ]
        
        for i, line in enumerate(ui_text):
            s = self.font.render(line, True, (100, 100, 100))
            self.screen.blit(s, (10, 10 + i * 20))

        # Equalizer Bars
        bar_width = 20
        start_x = 10
        start_y = CFG.HEIGHT - 10
        
        colors = [(255, 50, 50), (50, 255, 50), (50, 50, 255)]
        keys = ['bass', 'mid', 'treble']
        
        for i, key in enumerate(keys):
            h = energy[key] * 100
            pygame.draw.rect(self.screen, colors[i], 
                           (start_x + i * 25, start_y - h, bar_width, h))

    def run(self):
        while self.running:
            # 1. Event Handling
            mouse_pressed = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pressed = True
            
            mouse_state = pygame.mouse.get_pressed()[0] # Hold detection
            mouse_pos = pygame.mouse.get_pos()

            # 2. Data Acquisition
            energy = self.audio.get_energy()
            
            # 3. Physics Update
            self.physics.update(energy, mouse_pos, mouse_state)
            
            # 4. Rendering
            
            # A. Trail Effect: Draw a semi-transparent black box over the previous frame
            # instead of clearing it completely.
            self.trail_surface.fill((0, 0, 0)) # Black
            # Dynamic trail fade based on Treble (High energy = frantic strobe)
            fade_amt = 40 + int(energy['treble'] * 100) 
            self.trail_surface.set_alpha(fade_amt)
            self.screen.blit(self.trail_surface, (0, 0))
            
            # B. Draw Particles
            # We grab positions from numpy directly
            positions = self.physics.pos.astype(int)
            sizes = self.physics.attrs[:, 0]
            colors_idx = self.physics.colors
            
            # To optimize drawing, we lock the surface or use array blitting, 
            # but for < 5000 particles, circle drawing is fine.
            # Let's make the particles glow based on Bass
            radius_mult = 1.0 + (energy['bass'] * 2.0)
            
            for i in range(CFG.PARTICLE_COUNT):
                color = CFG.Palette[colors_idx[i]]
                # Modulate color brightness by Mid range
                bright_mod = max(1, int(energy['mid'] * 255))
                final_color = (
                    min(255, color[0] + bright_mod),
                    min(255, color[1] + bright_mod),
                    min(255, color[2] + bright_mod)
                )
                
                pos = positions[i]
                r = max(1, int(sizes[i] * radius_mult))
                
                pygame.draw.circle(self.screen, final_color, pos, r)
                
            # C. Draw Connections (Lattice)
            # Calculating all-pairs distance is too heavy (O(N^2)).
            # We only draw connections for a subset or use the mouse as a hub.
            # Let's draw lines from Mouse to nearby particles
            if mouse_state:
                m_pos = np.array(mouse_pos)
                dists = np.linalg.norm(self.physics.pos - m_pos, axis=1)
                close_indices = np.where(dists < 150)[0]
                
                for idx in close_indices:
                    start = m_pos.astype(int)
                    end = positions[idx]
                    # Line color based on Treble
                    line_col = (255, 255, 255) if energy['treble'] > 0.5 else (100, 100, 255)
                    pygame.draw.aaline(self.screen, line_col, start, end)

            self.draw_ui(energy)
            pygame.display.flip()
            self.clock.tick(CFG.FPS)

        # Cleanup
        self.audio.close()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = AetherialLattice()
    app.run()