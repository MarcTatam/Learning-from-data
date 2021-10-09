import csv
import os
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

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
    with open('understat_Bournemouth.csv') as csvfile:
        bournemouthfile = csv.reader(csvfile)
        for row in bournemouthfile:
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
    with open('understat_Norwich.csv') as csvfile:
        norwichfile = csv.reader(csvfile)
        for row in norwichfile:
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
    with open('understat_Watford.csv') as csvfile:
        watfordfile = csv.reader(csvfile)
        for row in watfordfile:
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
    ax.scatter(xG_win, xGA_win, c = 'green')
    ax.scatter(xG_loss, xGA_loss, c = 'red')
    #ax.scatter(xG_draw, xGA_draw)

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
    ax.scatter(npxG_win, npxGA_win, c = 'green')
    ax.scatter(npxG_loss, npxGA_loss, c = 'red')
    #ax.scatter(npxG_draw, xGA_draw)

    ax.set_xlabel('npxG for', fontsize=15)
    ax.set_ylabel('npxG against', fontsize=15)
    plt.show()

def graph_deep(master_dict:dict):
    datapoints = len(master_dict["xG"])
    deep_loss = []
    deep_win = []
    deep_draw = []

    deepA_loss = []
    deepA_win = []
    deepA_draw = []

    for i in range(0,datapoints):
        if master_dict["result"][i] == "w":
            deep_win.append(int(master_dict["deep"][i]))
            deepA_win.append(int(master_dict["deep_allowed"][i]))
        elif master_dict["result"][i] == "l":
            deep_loss.append(int(master_dict["deep"][i]))
            deepA_loss.append(int(master_dict["deep_allowed"][i]))
        else:
            deep_draw.append(int(master_dict["deep"][i]))
            deepA_draw.append(int(master_dict["deep_allowed"][i]))

    fig, ax = plt.subplots()
    ax.scatter(deep_win, deepA_win, c = 'green')
    ax.scatter(deep_loss, deepA_loss, c = 'red')
    #ax.scatter(deep_draw, deepA_draw)

    ax.set_xlabel('Deep plays for', fontsize=15)
    ax.set_ylabel('Deep plays against', fontsize=15)
    plt.show()

    
if __name__ == "__main__":
    master = import_data()
    graph_deep(master)