import numpy as np 
import math
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# Constants
x_hat = "x̂"  
y_hat = "ŷ" 

# Constrain Checks
def validate_2d_array(array, name):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")
    if len(array) != 2:
        raise ValueError(f"{name} must have exactly two dimensions.")

# Object Abstractions
class Object:
    def __init__(self, name: str, mass: float, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray):

        validate_2d_array(position, "Position")
        validate_2d_array(velocity, "Velocity")
        validate_2d_array(acceleration, "Acceleration")

        if mass <= 0:
            raise ValueError("Objects cannot be Massless.")

        self.mass = mass
        self.name = name
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.acceleration = np.array(acceleration, dtype=np.float64)

    def __str__(self):

        return (
            f"{self.name} with {self.mass} Kgs\n"
            f"Position: {self.position[0]}{x_hat} + {self.position[1]}{y_hat}\n"
            f"Velocity: {self.velocity[0]}{x_hat} + {self.velocity[1]}{y_hat}\n"
            f"Acceleration: {self.acceleration[0]}{x_hat} + {self.acceleration[1]}{y_hat}"
        )
    
    def update(self, dt: float, forces: list[np.ndarray]) -> None:

        if dt <= 0:
            raise ValueError("Time step dt must be greater than zero.")

        total_force = sum(forces, start=np.array([0.0, 0.0]))
        self.acceleration = total_force / self.mass
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

class Spring(Object):
    def __init__(self, name: str, mass: float, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray, k: float, eqib: np.ndarray):

        validate_2d_array(eqib, "Equilibrium Position")
        
        super().__init__(name, mass, position, velocity, acceleration)  
        self.k = k
        self.eqib = np.array(eqib, dtype=np.float64)


    def force(self, obj1: Object, obj2: Object = None) -> np.ndarray:
        if obj2 is None:  # Default to equilibrium position
            obj2_position = self.eqib
        else:
            obj2_position = obj2.position

        displacement = obj1.position - obj2_position
        if np.allclose(displacement, [0.0, 0.0]):
            return np.array([0.0, 0.0])
        return -self.k * displacement



class CoupledOscillators():
    def __init__(self, spring1: Spring, spring2: Spring, spring3: Spring, block1: object, block2: object):

        if not isinstance(block1, Object) or not isinstance(block2, Object):
            raise TypeError("block1 and block2 must be instances of Object.")
        if not all(isinstance(s, Spring) for s in [spring1, spring2, spring3]):
            raise TypeError("spring1, spring2, and spring3 must be instances of Spring.")

        self.spring1 = spring1
        self.spring2 = spring2
        self.spring3 = spring3
        self.block1  = block1
        self.block2  = block2
        

        m1, m2 = block1.mass, block2.mass
        k1, k2, k3 = spring1.k, spring2.k, spring3.k
        
        self.mass_matrix = np.array([[m1, 0], [0, m2]], dtype=np.float64)
        self.stiffness_matrix = np.array([[k1 + k2, -k2],
                                          [-k2, k2 + k3]], dtype=np.float64)
        

        try:
            self.eigenvalues, self.eigenvectors = np.linalg.eig(
                np.dot(np.linalg.inv(self.mass_matrix), self.stiffness_matrix)
            )
        except np.linalg.LinAlgError:
            raise ValueError("Matrix inversion failed, likely due to singular mass matrix.")
        
        
        if np.any(np.iscomplex(self.eigenvalues)):
            raise ValueError("System has complex eigenvalues, indicating instability or improper configuration.")
        
        
        self.omega_1, self.omega_2 = np.sqrt(self.eigenvalues.real)
        
        if self.omega_1 <= 0 or self.omega_2 <= 0:
            raise ValueError("One or more eigenfrequencies are non-positive, indicating an invalid physical system.")
    
    def compute_acceleration(self, block_index, t):
        
        mode_1 = self.eigenvectors[:, 0]
        mode_2 = self.eigenvectors[:, 1]
        
        a1 = -self.omega_1**2 * mode_1[block_index] * np.cos(self.omega_1 * t)
        a2 = -self.omega_2**2 * mode_2[block_index] * np.cos(self.omega_2 * t)
        
        return a1 + a2
        
        
    def simulate(self, total_time, time_step):
        
        time_array = np.arange(0, total_time, time_step)

        
        block1_positions = []
        block2_positions = []

        for t in time_array:
            force1 = self.spring1.force(self.block1) + self.spring2.force(self.block1, self.block2)
            force2 = self.spring3.force(self.block2) + self.spring2.force(self.block2, self.block1)

            self.block1.update(time_step, [force1])
            self.block2.update(time_step, [force2])

            block1_positions.append(self.block1.position[0])
            block2_positions.append(self.block2.position[0])

        return time_array, block1_positions, block2_positions

    def tweak_parameters(self, k2, k3, m2):

        self.spring2.k = k2
        self.spring3.k = k3
        self.block2.mass = m2
        

        m1, m2 = self.block1.mass, self.block2.mass
        k1, k2, k3 = self.spring1.k, self.spring2.k, self.spring3.k

        self.mass_matrix = np.array([[m1, 0], [0, m2]], dtype=np.float64)
        self.stiffness_matrix = np.array([[k1 + k2, -k2],
                                          [-k2, k2 + k3]], dtype=np.float64)

        try:
            self.eigenvalues, self.eigenvectors = np.linalg.eig(
                np.dot(np.linalg.inv(self.mass_matrix), self.stiffness_matrix)
            )
        except np.linalg.LinAlgError:
            raise ValueError("Matrix inversion failed, likely due to singular mass matrix.")
        
        if np.any(np.iscomplex(self.eigenvalues)):
            raise ValueError("System has complex eigenvalues, indicating instability or improper configuration.")
        
        self.omega_1, self.omega_2 = np.sqrt(self.eigenvalues.real)
        
        if self.omega_1 <= 0 or self.omega_2 <= 0:
            raise ValueError("One or more eigenfrequencies are non-positive, indicating an invalid physical system.")

    def generate_audio(self, duration=5, time_step=1/44100, filename="mass_1_audio.wav"):

        time_array, block1_positions, _ = self.simulate(duration, time_step)


        block1_audio = np.interp(block1_positions, (min(block1_positions), max(block1_positions)), (-1, 1))

        if np.ptp(block1_audio) < 0.01: 
            print("Warning: Oscillations are too small for audible sound. Check initial conditions.")
            block1_audio = block1_audio * 100   

        sampling_rate = int(1 / time_step)

        audio_signal = (block1_audio * 32767).astype(np.int16)

        write(filename, sampling_rate, audio_signal)
        print(f"Audio file saved as '{filename}'")

    def set_natural_frequency(self, target_frequency, mode=1, duration=5, time_step=1/44100, filename="mass_1_target_freq.wav", tolerance=1, max_iterations=100, max_adjustment=0.1):
        target_omega = 2 * np.pi * target_frequency
        iteration = 0

        original_mass1 = self.block1.mass
        original_mass2 = self.block2.mass

        while iteration < max_iterations:
            iteration += 1

            actual_omega_1, actual_omega_2 = self.omega_1, self.omega_2
            actual_frequencies = (actual_omega_1 / (2 * np.pi), actual_omega_2 / (2 * np.pi))

            if abs(actual_frequencies[mode - 1] - target_frequency) <= tolerance:
                print(f"Target frequency matched: Mode {mode}: {actual_frequencies[mode - 1]:.2f} Hz")
                break

            try:
                if mode == 1:
                    scaling_factor = min(max_adjustment, max(-max_adjustment, (target_omega / actual_omega_1)**2 - 1))
                    self.block1.mass *= 1 + scaling_factor
                elif mode == 2:
                    scaling_factor = min(max_adjustment, max(-max_adjustment, (target_omega / actual_omega_2)**2 - 1))
                    self.block2.mass *= 1 + scaling_factor
                else:
                    raise ValueError("Invalid mode. Choose mode 1 or 2.")

                self.tweak_parameters(self.spring2.k, self.spring3.k, self.block2.mass)

                if self.omega_1 <= 0 or self.omega_2 <= 0:
                    raise ValueError("Invalid eigenfrequencies encountered.")

            except (ValueError, OverflowError):
                self.block1.mass = original_mass1
                self.block2.mass = original_mass2
                print("Invalid system configuration during adjustment. Rolling back.")
                break

        else:
            print("Warning: Maximum iterations reached. Unable to match target frequency closely.")

        actual_omega_1, actual_omega_2 = self.omega_1, self.omega_2
        actual_frequencies = (actual_omega_1 / (2 * np.pi), actual_omega_2 / (2 * np.pi))
        print(f"Adjusted system frequencies: Mode 1: {actual_frequencies[0]:.2f} Hz, Mode 2: {actual_frequencies[1]:.2f} Hz")

        self.generate_audio(duration, time_step, filename)

# Initialize springs and blocks
spring1 = Spring("Spring1", 1.0, np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), 10.0, np.array([0, 0]))
spring2 = Spring("Spring2", 1.0, np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), 15.0, np.array([0, 0]))
spring3 = Spring("Spring3", 1.0, np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), 20.0, np.array([0, 0]))

block1 = Object("Block1", 2.0, np.array([1, 0]), np.array([0, 0]), np.array([0, 0]))
block2 = Object("Block2", 1.5, np.array([-1, 0]), np.array([0, 0]), np.array([0, 0]))

# Create the coupled system
coupled_system = CoupledOscillators(spring1, spring2, spring3, block1, block2)

# Simulation parameters
time_step = 0.01  # Time step for the simulation (seconds)
total_time = 10   # Total simulation time (seconds)

# Run the simulation
time_array, block1_positions, block2_positions = coupled_system.simulate(total_time, time_step)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time_array, block1_positions, label="Block 1 Position")
plt.plot(time_array, block2_positions, label="Block 2 Position")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Motion of Two Coupled Oscillators")
plt.legend()
plt.grid()
plt.show()


# Adjust initial conditions for meaningful oscillations
block1 = Object("Block1", 2.0, np.array([1.0, 0]), np.array([0, 0]), np.array([0, 0]))
block2 = Object("Block2", 1.5, np.array([-1.0, 0]), np.array([0, 0]), np.array([0, 0]))

# Recreate the coupled system
coupled_system = CoupledOscillators(spring1, spring2, spring3, block1, block2)

# Generate a 5-second audio file with some target subfrequncy
coupled_system.set_natural_frequency(
    target_frequency=0.2, 
    mode=2, 
    duration=5, 
    time_step=1/44100, 
    filename="mass_1_420Hz_mode2.wav"
)

