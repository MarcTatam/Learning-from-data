import csv
import os

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

if __name__ == "__main__":
    import_data()