import csv
import json
import os
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import numpy as np

def import_data()->dict:
    master_dict = {"xG" : [],
                   "xGA" : [],
                   "npxG" : [],
                   "npxGA" : [],
                   "ppda" : [],
                   "ppda_allowed" : [],
                   "deep": [],
                   "deep_allowed" : [],
                   "result" : []}
    with open('understat_Arsenal.csv') as csvfile:
        arsenalfile = csv.reader(csvfile)
        for row in arsenalfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Aston_Villa.csv') as csvfile:
        villafile = csv.reader(csvfile)
        for row in villafile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Brighton.csv') as csvfile:
        brightonfile = csv.reader(csvfile)
        for row in brightonfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Burnley.csv') as csvfile:
        burnleyfile = csv.reader(csvfile)
        for row in burnleyfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Chelsea.csv') as csvfile:
        chelseafile = csv.reader(csvfile)
        for row in chelseafile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Crystal_Palace.csv') as csvfile:
        palacefile = csv.reader(csvfile)
        for row in palacefile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Everton.csv') as csvfile:
        evertonfile = csv.reader(csvfile)
        for row in evertonfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Fulham.csv') as csvfile:
        fulhamfile = csv.reader(csvfile)
        for row in fulhamfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Leeds.csv') as csvfile:
        leedsfile = csv.reader(csvfile)
        for row in leedsfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Leicester.csv') as csvfile:
        leicesterfile = csv.reader(csvfile)
        for row in leicesterfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Liverpool.csv') as csvfile:
        liverpoolfile = csv.reader(csvfile)
        for row in liverpoolfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Manchester_City.csv') as csvfile:
        cityfile = csv.reader(csvfile)
        for row in cityfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Manchester_United.csv') as csvfile:
        unitedfile = csv.reader(csvfile)
        for row in unitedfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Newcastle_United.csv') as csvfile:
        newcastlefile = csv.reader(csvfile)
        for row in newcastlefile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Sheffield_United.csv') as csvfile:
        sheffieldfile = csv.reader(csvfile)
        for row in sheffieldfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Southampton.csv') as csvfile:
        southamptonfile = csv.reader(csvfile)
        for row in southamptonfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Tottenham.csv') as csvfile:
        spursfile = csv.reader(csvfile)
        for row in spursfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_West_Bromwich_Albion.csv') as csvfile:
        westbromfile = csv.reader(csvfile)
        for row in westbromfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_West_Ham.csv') as csvfile:
        westhamfile = csv.reader(csvfile)
        for row in westhamfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    with open('understat_Wolverhampton_Wanderers.csv') as csvfile:
        wolvesfile = csv.reader(csvfile)
        for row in wolvesfile:
            if row[0] == "h" and row[0] != "h_a":
                master_dict["xG"].append(row[1])
                master_dict["xGA"].append(row[2])
                master_dict["npxG"].append(row[3])
                master_dict["npxGA"].append(row[4])
                master_dict["ppda"].append(row[5])
                master_dict["ppda_allowed"].append(row[6])
                master_dict["deep"].append(row[7])
                master_dict["deep_allowed"].append(row[8])
                master_dict["result"].append(row[12])
    return master_dict

def graph_xG(master_dict:dict):
    datapoints = len(master_dict["xG"])
    xG_loss = []
    xG_win = []
    xG_draw = []

    xGA_loss = []
    xGA_win = []
    xGA_draw = []

    for i in range(0,datapoints):
        if master_dict["result"][i] == "w":
            xG_win.append(float(master_dict["xG"][i]))
            xGA_win.append(float(master_dict["xGA"][i]))
        elif master_dict["result"][i] == "l":
            xG_loss.append(float(master_dict["xG"][i]))
            xGA_loss.append(float(master_dict["xGA"][i]))
        else:
            xG_draw.append(float(master_dict["xG"][i]))
            xGA_draw.append(float(master_dict["xGA"][i]))

    fig, ax = plt.subplots()
    scatterw = ax.scatter(xG_win, xGA_win, c = 'green')
    scatterl = ax.scatter(xG_loss, xGA_loss, c = 'red')
    scatterd = ax.scatter(xG_draw, xGA_draw, c = 'orange')

    ax.legend([scatterw, scatterl, scatterd], ["Win", "Loss", "Draw"])

    ax.set_xlabel('xG for', fontsize=15)
    ax.set_ylabel('xG against', fontsize=15)
    plt.show()

def graph_npxG(master_dict:dict):
    datapoints = len(master_dict["xG"])
    npxG_loss = []
    npxG_win = []
    npxG_draw = []

    npxGA_loss = []
    npxGA_win = []
    npxGA_draw = []

    for i in range(0,datapoints):
        if master_dict["result"][i] == "w":
            npxG_win.append(float(master_dict["npxG"][i]))
            npxGA_win.append(float(master_dict["npxGA"][i]))
        elif master_dict["result"][i] == "l":
            npxG_loss.append(float(master_dict["npxG"][i]))
            npxGA_loss.append(float(master_dict["npxGA"][i]))
        else:
            npxG_draw.append(float(master_dict["npxG"][i]))
            npxGA_draw.append(float(master_dict["npxGA"][i]))

    fig, ax = plt.subplots()
    scatterw = ax.scatter(npxG_win, npxGA_win, c = 'green')
    scatterl = ax.scatter(npxG_loss, npxGA_loss, c = 'red')
    #ax.scatter(npxG_draw, xGA_draw)

    ax.legend([scatterw, scatterl], ["Win", "Loss"])

    ax.set_xlabel('npxG for', fontsize=15)
    ax.set_ylabel('npxG against', fontsize=15)
    plt.show()

def graph_deep(master_dict:dict):
    datapoints = len(master_dict["xG"])

    deep_draw = []


    deepA_draw = []

    matrixw = np.zeros((24,22))
    matrixl = np.zeros((24,22))

    for i in range(0,datapoints):
        if master_dict["result"][i] == "w":
            matrixw[int(master_dict["deep"][i])][int(master_dict["deep_allowed"][i])] += 1
        elif master_dict["result"][i] == "l":
            matrixl[int(master_dict["deep"][i])][int(master_dict["deep_allowed"][i])] += 1
        else:
            deep_draw.append(int(master_dict["deep"][i]))
            deepA_draw.append(int(master_dict["deep_allowed"][i]))
    fig, axes = plt.subplots(ncols=2)
    plt.subplots_adjust(wspace=0.25)
    ax1, ax2 = axes
    im1 = ax1.imshow(matrixw, origin = 'lower')
    im2 = ax2.imshow(matrixl, origin = 'lower')
    
    ax1.set_title('Win')
    ax2.set_title('Loss')
    ax1.set_xlabel('Deep Allowed')
    ax2.set_xlabel('Deep Allowed')
    ax1.set_ylabel('Deep')
    ax2.set_ylabel('Deep')

    plt.colorbar(im1, ax = ax1, shrink = 0.6)
    plt.colorbar(im2, ax = ax2, shrink = 0.6)
    plt.show()

def graph_poss(master_dict:dict):
    datapoints = len(master_dict["xG"])
    poss_loss = []
    poss_win = []
    poss_draw = []

    possA_loss = []
    possA_win = []
    possA_draw = []

    for i in range(0,datapoints):
        parsed_format = json.loads(master_dict["ppda"][i].replace("'","\""))
        parsed_formatA = json.loads(master_dict["ppda_allowed"][i].replace("'","\""))
        if master_dict["result"][i] == "w":
            poss_win.append(parsed_format["att"])
            possA_win.append(parsed_formatA["att"])
        elif master_dict["result"][i] == "l":
            poss_loss.append(parsed_format["att"])
            possA_loss.append(parsed_formatA["att"])
        else:
            poss_draw.append(parsed_format["att"])
            possA_draw.append(parsed_formatA["att"])

    fig, ax = plt.subplots()
    scatterw = ax.scatter(poss_win, possA_win, c = 'green')
    scatterl = ax.scatter(poss_loss, possA_loss, c = 'red')
   

    ax.legend([scatterw, scatterl], ["Win", "Loss"])

    ax.set_xlabel('Away team passes', fontsize=15)
    ax.set_ylabel('Home team passes', fontsize=15)
    plt.show()

def graph_ppda(master_dict:dict):
    datapoints = len(master_dict["xG"])
    ppda_loss = []
    ppda_win = []
    ppda_draw = []

    ppdaA_loss = []
    ppdaA_win = []
    ppdaA_draw = []

    for i in range(0,datapoints):
        parsed_format = json.loads(master_dict["ppda"][i].replace("'","\""))
        parsed_formatA = json.loads(master_dict["ppda_allowed"][i].replace("'","\""))
        if master_dict["result"][i] == "w":
            ppda_win.append(parsed_format["att"]/parsed_format["def"])
            ppdaA_win.append(parsed_formatA["att"]/parsed_formatA["def"])
        elif master_dict["result"][i] == "l":
            ppda_loss.append(parsed_format["att"]/parsed_format["def"])
            ppdaA_loss.append(parsed_formatA["att"]/parsed_formatA["def"])
        else:
            ppda_draw.append(parsed_format["att"]/parsed_format["def"])
            ppdaA_draw.append(parsed_formatA["att"]/parsed_formatA["def"])

    fig, ax = plt.subplots()
    scatterw = ax.scatter(ppda_win, ppdaA_win, c = 'green')
    scatterl = ax.scatter(ppda_loss, ppdaA_loss, c = 'red')
    #scatterd, = ax.scatter(deep_draw, deepA_draw)
   

    ax.legend([scatterw, scatterl], ["Win", "Loss"])

    ax.set_xlabel('Home team PPDA', fontsize=15)
    ax.set_ylabel('Away team PPDA', fontsize=15)
    plt.show()

    
if __name__ == "__main__":
    master = import_data()
    graph_xG(master)