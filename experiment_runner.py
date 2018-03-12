import importlib
import run_agent
import time
import actr

run_agent = importlib.import_module('run_agent')

for x in range(0,3):
    run_agent.file_name_count = x
    t = run_agent.main()
    #run_agent.agent.actr.stop()


    time.sleep(5)
    run_agent.agent.actr.reset()
    #run_agent.agent.actr = actr
    #del run_agent.agent.actr = actr
    #del run_agent


    #run_agent = importlib.import_module('run_agent')

print("complete")