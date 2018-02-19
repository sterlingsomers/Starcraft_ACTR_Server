import run_agent


#for the graphs



import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import threading


#style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_ylim([0,1])

#fig2 = plt.figure()
ax2 = fig.add_subplot(212)







def animate(i):
    print("agent", run_agent.agent)
    if not run_agent.agent:
        return
    history = run_agent.agent.history
    chks = run_agent.agent.dict_dm

    xs1 = []
    ys1 = []



    xs2 = [1,2,3,4,5]
    ys2 = [0,0,0,0,0] #data values
    count = [0,0,0,0,0]

    # if chks[('TRUE','FALSE','FALSE','SELECT-BEACON')]:
    #     xs.append('green-only')
    # if chks[('FALSE','TRUE','FALSE','SELECT-BEACON')]:
    #     xs.append('orange-only')
    # if chks[('TRUE','TRUE','FALSE','SELECT-BEACON')]:
    #     xs.append('not-blocking')
    # if chks[('TRUE','TRUE','TRUE','SELECT-AROUND')]:
    #     xs.append('blocking-go-around')
    #if chks[('TRUE','TRUE','TRUE','SELECT-BEACON')]:
    #    xs.append(5)
    #
    num = len(history)
    for case in history:
        if case['green'] and not case['orange'] and not case['blocking']:
            count[0] += 1
            ys2[0] = count[0] / num
        if not case['green'] and case['orange'] and not case['blocking']:
            count[1] += 1
            ys2[1] = count[1] / num
        if case['green'] and case['orange'] and not case['blocking']:
            count[2] += 1
            ys2[2] = count[2] / num
        if case['green'] and case['orange'] and case['blocking']:
            if 'actr' in case:
                if case['actr'] == case['chosen_action']:
                    xs1.append(1)
                else:
                    xs1.append(0)

                if case['actr'] == 'SELECT_AROUND':
                    count[3] += 1
                    ys2[3] = count[3] / num
                else:
                    count[4] += 1
                    ys2[4] = count[4] / num
            else:
                pass


    ax2.clear()
    ax2.bar(xs2,ys2)

    sum = 0.0
    numb = 0.0
    print("xs1",xs1, len(xs1))
    xps = []
    yps = []
    for x in xs1:
        numb += 1
        sum += x
        xps.append(sum / numb)
        yps.append(numb)

    ax1.clear()
    ax1.plot(yps, xps)




    # if not run_agent.agent:
    #     ax1.clear()
    #     ax1.plot([0,1,2], [0,5,7])
    #     return
    # if not run_agent.agent.history:
    #     ax1.clear()
    #     ax1.plot([1], [1])
    #     return

    # history = run_agent.agent.history
    # xs = []
    # ys = []



    # for case in history:
    #     if case['green'] and case['orange'] and case['blocking']:
    #         if case['actr'] == case['chosen_action']:
    #             xs.append(1)
    #         else:
    #             xs.append(0)







if __name__ == "__main__":
    run_thread = threading.Thread(target=run_agent.main)
    run_thread.daemon = True
    run_thread.start()

    print("thread started")
    ani = animation.FuncAnimation(fig, animate, interval=500)
    plt.show()


