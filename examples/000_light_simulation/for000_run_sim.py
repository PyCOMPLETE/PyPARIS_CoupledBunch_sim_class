from PyPARIS.ring_of_CPUs_multiturn import RingOfCPUs_multiturn
import Simulation 

simulation_content = Simulation.Simulation()
myCPUring = RingOfCPUs_multiturn(simulation_content)
myCPUring.run()
