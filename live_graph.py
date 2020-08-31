import run_agent


#for the graphs

import numpy as np

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

import threading


#style.use('fivethirtyeight') g
fig, (ax1,ax3,ax4,ax5) = plt.subplots(4)
plt.tight_layout()
#ax1.title = "Action Selection"
#fig = plt.figure()
#ax1 = fig.add_subplot(211)
#ax1.set_ylim(bottom=0,top=1,auto=False)
#ax1.set_xlim([0,1],auto=False)

#fig2 = plt.figure()
#ax2 = fig.add_subplot(212)


#ax3 = fig.add_subplot(3,3,3)





def animate(i):
    ###print("agent", run_agent.agent)
    if not run_agent.agent:
        return
    # history = run_agent.agent.history
    # chks = run_agent.agent.dict_dm

    xs1 = []
    ys1 = []



    xs2 = ["select-green","select-orange","select-around"]
    ys2 = run_agent.agent.blend_values
    if not ys2:
        ys2 = [0,0,0]
    count = [0,0,0,0,0]

    case = ''
    text = ""
    # if history:
    #     case = history[-1]
    #     # if case['green'] and not case['orange'] and not case['blocking'] and case['chosen_action'] == 'SELECT_BEACON':
    #     #     text = 'select-green'
    #     # if not case['green'] and case['orange'] and not case['blocking'] and case['chosen_action'] == 'SELECT_BEACON':
    #     #     text='select-orange'
    #     ###print("ASDFASFADSFADSFAFDSFDAFASFADSF", case)
    #     if case['green'] and not case['orange'] and not case['blocking']:
    #         text = 'select-green'
    #     if not case['green'] and case['orange'] and not case['blocking']:
    #         text = 'select-orange'
    #     if case['green'] and case['orange'] and not case['blocking']:
    #         text = 'not-blocking'
    #     if case['green'] and case['orange'] and case['blocking']:
    #         text = 'not-blocking'
    # num = len(history)
    # for case in history:
    #     if case['green'] and not case['orange'] and not case['blocking']:
    #         count[0] += 1
    #         ys2[0] = count[0] / num
    #     if not case['green'] and case['orange'] and not case['blocking']:
    #         count[1] += 1
    #         ys2[1] = count[1] / num
    #     if case['green'] and case['orange'] and not case['blocking']:
    #         count[2] += 1
    #         ys2[2] = count[2] / num
    #     if case['green'] and case['orange'] and case['blocking']:
    #         if 'actr' in case:
    #             if case['actr'] == case['chosen_action']:
    #                 xs1.append(1)
    #             else:
    #                 xs1.append(0)
    #
    #             if case['actr'] == 'SELECT_AROUND':
    #                 count[3] += 1
    #                 ys2[3] = count[3] / num
    #             else:
    #                 count[4] += 1
    #                 ys2[4] = count[4] / num
    #         else:
    #             pass

    ax1.clear()
    # ax2.text(text)
    #ax2.clear()
    ax1.bar(xs2,ys2)

    su = 0.0
    numb = 0.0
    ###print("xs1",xs1, len(xs1))
    # xps = []
    # yps = []
    # for x in xs1:
    #     numb += 1
    #     sum += x
    #     xps.append(sum / numb)
    #     yps.append(numb)
    #
    # ax1.clear()
    # ax1.plot(yps, xps)


    # if type(run_agent.agent.fc1) == int:
    #     return


    #Hide the network activity graph for now
    #commented out below.
    # #print("agent",run_agent.agent.fc1)
    # fc1 = np.array(run_agent.agent.fc1)
    # fc1 = fc1.reshape((16,16))
    # #print("fc1", fc1,fc1.shape)
    #
    # ax3.clear()
    # ax3.matshow(fc1)



    #print("fc1",run_agent.agent.fc1)
#        run_agent.agent.fc1 = run_agent.agent.fc1.reshape((16,16))

    # #NEED some way to get the salience and plot it
    # #dummy plot
    # #
    categories = ('Green', 'Orange', 'Blocking')
    y_pos = np.arange(len(categories))
    # saliences = run_agent.agent.saliences
    # salience = []
    # green, orange, blocking = [], [], []
    # for key in saliences:
    #     green.append(saliences[key][0])
    #     orange.append(saliences[key][1])
    #     blocking.append(saliences[key][2])
    #
    # salience = [sum(green),sum(orange),sum(blocking)]
    # if not salience:
    #     salience = [0,0,0]
    # ###print("saliences", salience)
    # ax3.clear()
    # #ax3.set_xlim([0,1.0])
    # #ax3.set_title(text)
    # ax3.barh(y_pos, salience, align='center')
    # ax3.set_yticks(y_pos)
    # ax3.set_yticklabels(['green','orange','blocking'])
    # ax3.invert_yaxis()
    # ax3.set_xlabel('Salience')
    all_axies = [ax3,ax4,ax5]
    saliences = run_agent.agent.saliences
    if not saliences:
        return
    for key,graph in zip(saliences,all_axies):
        graph.clear()
        graph.set_title(key)
        graph.barh(y_pos,saliences[key],align='center')
        graph.set_yticks(y_pos)
        graph.set_yticklabels(['green','orange','blocking'])
        graph.invert_yaxis()
    ax5.set_xlabel('Salience')
    plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 1.0)











if __name__ == "__main__":
    run_thread = threading.Thread(target=run_agent.main)
    run_thread.daemon = True
    run_thread.start()

    print("thread started")
    ani = animation.FuncAnimation(fig, animate, interval=200)
    plt.show()


