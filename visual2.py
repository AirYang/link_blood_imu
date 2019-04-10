# -*- coding: utf-8 -*-

import ast
import csv
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


def read_data(cuffpath, personpath, sensorpath):

    result = []

    with open("data/person.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            result.append({
                "age": int(row["age"]),
                "group": int(row["group"]),
                "weight": float(row["weight"]),
                "height": float(row["height"]),
                "cuff": [],
                "sensor": []
            })

    with open("data/cuff.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            result[len(result) - int(row["group"])]["cuff"].append({
                "time": row["time"],
                "pulserate": int(row["pulserate"]),
                "systolicbloodpressure": int(row["systolicbloodpressure"]),
                "diastolicbloodpressure": int(row["diastolicbloodpressure"])
            })

    with open("data/sensor.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            if(int(row["pulserate"]) == 0) or (int(row["pulserate"]) == 255) or (int(row["systolicbloodpressure"]) == 0) or (int(row["systolicbloodpressure"]) == 255) or (int(row["diastolicbloodpressure"]) == 0) or (int(row["diastolicbloodpressure"]) == 255):
                continue

            result[len(result) - int(row["group"])]["sensor"].append({
                "time": row["time"],
                "pulserate": int(row["pulserate"]),
                "systolicbloodpressure": int(row["systolicbloodpressure"]),
                "diastolicbloodpressure": int(row["diastolicbloodpressure"]),
                "euler": ast.literal_eval(row["euler"]),
                "quaternion": ast.literal_eval(row["quaternion"]),
                "acceleration": ast.literal_eval(row["acceleration"]),
            })

    return result


def show_data(datalist):

    originalerrors = [
        [],  # for pt
        [],  # for sbp
        [],  # for dbp
        []   # for id
    ]

    for group in datalist:

        cuffshow = {
            "time": [],
            "pulserate": [],
            "systolicbloodpressure": [],
            "diastolicbloodpressure": []
        }

        for elem in group["cuff"]:
            cuffshow["time"].append(time.mktime(time.strptime(
                elem["time"][0:20], "%Y-%m-%dT%H:%M:%S.")) + float(elem["time"][19:23]))
            cuffshow["pulserate"].append(elem["pulserate"])
            cuffshow["systolicbloodpressure"].append(
                elem["systolicbloodpressure"])
            cuffshow["diastolicbloodpressure"].append(
                elem["diastolicbloodpressure"])

        sensorshow = {
            "time": [],
            "pulserate": [],
            "systolicbloodpressure": [],
            "diastolicbloodpressure": []
        }

        for elem in group["sensor"]:
            sensorshow["time"].append(time.mktime(time.strptime(
                elem["time"][0:20], "%Y-%m-%dT%H:%M:%S.")) + float(elem["time"][19:23]))
            sensorshow["pulserate"].append(elem["pulserate"])
            sensorshow["systolicbloodpressure"].append(
                elem["systolicbloodpressure"])
            sensorshow["diastolicbloodpressure"].append(
                elem["diastolicbloodpressure"])

        pulserateerror = 0
        for elem in sensorshow["pulserate"][-10:]:
            pulserateerror += abs(elem-cuffshow["pulserate"][-1])
        pulserateerror /= 10

        systolicbperror = 0
        for elem in sensorshow["systolicbloodpressure"][-10:]:
            systolicbperror += abs(elem-cuffshow["systolicbloodpressure"][-1])
        systolicbperror /= 10

        diastolicerror = 0
        for elem in sensorshow["diastolicbloodpressure"][-10:]:
            diastolicerror += abs(elem-cuffshow["diastolicbloodpressure"][-1])
        diastolicerror /= 10

        print("raw data", group["group"], format(pulserateerror, ">5.2f"), format(
            systolicbperror, ">5.2f"), format(diastolicerror, ">5.2f"))

        # originalerrors.append([format(pulserateerror, ">5.2f"), format(
        #     systolicbperror, ">5.2f"), format(diastolicerror, ">5.2f")])

        originalerrors[0].append(round(pulserateerror, 2))
        originalerrors[1].append(round(systolicbperror, 2))
        originalerrors[2].append(round(diastolicerror, 2))
        originalerrors[3].append(group["group"])

        # print(sensorshow["pulserate"][-10:], sensorshow["systolicbloodpressure"]
        #       [-10:], sensorshow["diastolicbloodpressure"][-10:])
        # print(cuffshow["pulserate"][-1], cuffshow["systolicbloodpressure"]
        #       [-1], cuffshow["diastolicbloodpressure"][-1])

        imushow = {
            "time": [],
            "speedabsolute": [],
            "anglechange": [],
            "accelerationabsolute": []
        }

        imuspeed = [0, 0, 0]
        prequaternion = None
        for elem in group["sensor"]:
            imushow["time"].append(time.mktime(time.strptime(
                elem["time"][0:20], "%Y-%m-%dT%H:%M:%S.")) + float(elem["time"][19:23]))
            imushow["accelerationabsolute"].append(math.sqrt(pow(
                elem["acceleration"][0], 2) + pow(elem["acceleration"][1], 2) + pow(elem["acceleration"][2], 2))/3)
            imuspeed[0] = imuspeed[0] + round(elem["acceleration"][0], 1)
            imuspeed[1] = imuspeed[1] + round(elem["acceleration"][1], 1)
            imuspeed[2] = imuspeed[2] + round(elem["acceleration"][2], 1)
            imushow["speedabsolute"].append(math.sqrt(
                pow(imuspeed[0], 2) + pow(imuspeed[1], 2) + pow(imuspeed[2], 2)
            ))
            if prequaternion == None:
                imushow["anglechange"].append(0)
            else:
                pdot = prequaternion[0]*elem["quaternion"][0] + prequaternion[1]*elem["quaternion"][1] + \
                    prequaternion[2]*elem["quaternion"][2] + \
                    prequaternion[3]*elem["quaternion"][3]
                imushow["anglechange"].append(
                    math.degrees(math.acos(round(pdot, 6))))
            prequaternion = elem["quaternion"]

        # # for bp&pt show
        # fig = plt.figure(num="group-"+str(group["group"]), figsize=[15, 10])

        # pulserateplot = fig.add_subplot(211)

        # pulserateplot.scatter(sensorshow["time"],
        #                       sensorshow["pulserate"], s=10, label="$sensor$", c="blue")
        # pulserateplot.legend(ncol=2)

        # pulserateplot.scatter(cuffshow["time"], cuffshow["pulserate"], s=30,
        #                       label="$cuff$", c="red", edgecolors="black")
        # pulserateplot.legend(ncol=2)

        # # pulserateplot.grid(True)
        # pulserateplot.set_xlabel("$time$")
        # pulserateplot.set_ylabel("$bpm$")
        # pulserateplot.set_title(
        #     "pulserate")

        # bloodpressureplot = fig.add_subplot(212)

        # bloodpressureplot.scatter(sensorshow["time"],
        #                           sensorshow["systolicbloodpressure"], s=10, label="$sensor.systolic$", c="blue")
        # bloodpressureplot.legend(ncol=2)

        # bloodpressureplot.scatter(sensorshow["time"],
        #                           sensorshow["diastolicbloodpressure"], s=10, label="$sensor.diastolic$", c="green")
        # bloodpressureplot.legend(ncol=2)

        # bloodpressureplot.scatter(cuffshow["time"], cuffshow["systolicbloodpressure"], s=30,
        #                           label="$cuff.systolic$", c="red", edgecolors="black")
        # bloodpressureplot.legend(ncol=2)

        # bloodpressureplot.scatter(cuffshow["time"], cuffshow["diastolicbloodpressure"], s=30,
        #                           label="$cuff.diastolic$", c="pink", edgecolors="black")
        # bloodpressureplot.legend(ncol=2)

        # # bloodpressureplot.grid(True)
        # bloodpressureplot.set_xlabel("$time$")
        # bloodpressureplot.set_ylabel("$mmHg$")
        # bloodpressureplot.set_title(
        #     "bloodpressure")

        # plt.tight_layout()
        # plt.subplots_adjust(hspace=0.3)
        # plt.savefig("image/group"+str(group["group"]) + "_bp&pr.jpg")
        # # plt.show()

        # # for imu show
        # fig = plt.figure(num="group-"+str(group["group"]), figsize=[10, 12])

        # accplot = fig.add_subplot(211)

        # accplot.scatter(imushow["time"],
        #                 imushow["accelerationabsolute"], s=10, label="$acc$", c="blue")
        # accplot.legend()

        # accplot.set_xlabel("$time$")
        # accplot.set_title("acceleration absolute")

        # angleplot = fig.add_subplot(212)

        # angleplot.scatter(imushow["time"],
        #                 imushow["anglechange"], s=10, label="$ang$", c="green")
        # angleplot.legend()

        # angleplot.set_xlabel("$time$")
        # angleplot.set_title("angle change")

        # angleplot = fig.add_subplot(313)

        # angleplot.scatter(imushow["time"],
        #                 imushow["speedabsolute"], s=10, label="$spe$", c="green")
        # angleplot.legend()

        # angleplot.set_xlabel("$time$")
        # angleplot.set_title("speed absolute")

        # plt.tight_layout()
        # plt.subplots_adjust(hspace=0.3)
        # plt.savefig("image/group"+str(group["group"]) + "_imu.jpg")
        # # # plt.show()
        # plt.close()

    originalerrors[0].reverse()
    originalerrors[1].reverse()
    originalerrors[2].reverse()
    originalerrors[3].reverse()

    # # for error bar
    # fig = plt.figure(num="original-error", figsize=[18, 5])
    # # ids = [str(x) for x in range(len(datalist))]

    # x = list(range(len(datalist)))
    # total_width, n = 0.9, 3
    # width = total_width / n
    # # x = x - (total_width - width) / 2

    # errplot = fig.add_subplot(111)
    # errplot.bar(x, originalerrors[0], width=width, label="pulserate")

    # errplot.bar([c+width for c in x], originalerrors[1], width=width, label="systolicbp", tick_label=originalerrors[3])

    # errplot.bar([c+width*2 for c in x], originalerrors[2], width=width, label="diastolicbp")
    # errplot.legend()

    # errplot.set_xlabel("$group.id$")
    # errplot.set_title("original error")

    # plt.tight_layout()
    # plt.subplots_adjust(hspace=0.3)
    # plt.savefig("image/originalerror.jpg")
    # # plt.show()

    return originalerrors


def train_data(datalist):

     # predict errors
    mlperrors = [
        [],  # for pt
        [],  # for sbp
        [],  # for dbp
        []   # for id
    ]

    bayesianerrors = [
        [],  # for pt
        [],  # for sbp
        [],  # for dbp
        []   # for id
    ]

    bayesianpred = [
        [], # for pt
        [], # for sbp
        [], # for dbp
        []  # for id
    ]

    adaboosterrors = [
        [],  # for pt
        [],  # for sbp
        [],  # for dbp
        []   # for id
    ]

    knnerrors = [
        [],  # for pt
        [],  # for sbp
        [],  # for dbp
        []   # for id
    ]

    gradienterrors = [
        [],  # for pt
        [],  # for sbp
        [],  # for dbp
        []   # for id
    ]

    svmerrors = [
        [],  # for pt
        [],  # for sbp
        [],  # for dbp
        []   # for id
    ]

    res = [
        [], # for pt
        [], # for sbp
        [], # for dbp
        []  # for id
    ]

    for testindex in range(len(datalist)):

        groupid = datalist[testindex]["group"]
        train_x = []
        train_y = []
        test_x = []
        test_y = []

        # for single train
        train_y_pt = []
        train_y_sbp = []
        train_y_dbp = []

        # for single train mlp
        # train_y_pt_list = []
        # train_y_sbp_list = []
        # train_y_dbp_list = []

        # predict errors
        # predicterrors[0] = []
        # predicterrors[1] = []
        # predicterrors[2] = []

        for index in range(len(datalist)):
            group = datalist[index]
            if index == testindex:
                test_y.append([group["cuff"][1]["pulserate"], group["cuff"][1]
                               ["systolicbloodpressure"], group["cuff"][1]["diastolicbloodpressure"]])
                # test_x.append([])
                test_unit = []
                prequaternion = None
                for elem in group["sensor"][-60:]:
                    # print(elem["pulserate"], elem["systolicbloodpressure"], elem["diastolicbloodpressure"])
                    # print(elem["euler"], elem["quaternion"], elem["acceleration"])

                    test_unit.extend(
                        [elem["pulserate"], elem["systolicbloodpressure"], elem["diastolicbloodpressure"]])

                    test_unit.append(math.sqrt(pow(elem["acceleration"][0], 2) + pow(
                        elem["acceleration"][1], 2) + pow(elem["acceleration"][2], 2))/3)

                    if prequaternion == None:
                        test_unit.append(0)
                    else:
                        pdot = prequaternion[0]*elem["quaternion"][0] + prequaternion[1]*elem["quaternion"][1] + \
                            prequaternion[2]*elem["quaternion"][2] + \
                            prequaternion[3]*elem["quaternion"][3]
                        test_unit.append(math.degrees(
                            math.acos(round(pdot, 6))))
                    prequaternion = elem["quaternion"]

                test_x.append(test_unit)

            else:
                train_y.append([group["cuff"][1]["pulserate"], group["cuff"][1]
                                ["systolicbloodpressure"], group["cuff"][1]["diastolicbloodpressure"]])
                # train_x.append([])
                train_y_pt.append(group["cuff"][1]["pulserate"])
                train_y_sbp.append(group["cuff"][1]["systolicbloodpressure"])
                train_y_dbp.append(group["cuff"][1]["diastolicbloodpressure"])

                # train_y_pt_list.append([group["cuff"][1]["pulserate"]])
                # train_y_sbp_list.append(
                #     [group["cuff"][1]["systolicbloodpressure"]])
                # train_y_dbp_list.append(
                #     [group["cuff"][1]["diastolicbloodpressure"]])

                train_unit = []
                prequaternion = None
                for elem in group["sensor"][-60:]:
                    # print(elem["pulserate"], elem["systolicbloodpressure"], elem["diastolicbloodpressure"])
                    # print(elem["euler"], elem["quaternion"], elem["acceleration"])

                    train_unit.extend(
                        [elem["pulserate"], elem["systolicbloodpressure"], elem["diastolicbloodpressure"]])

                    train_unit.append(math.sqrt(pow(elem["acceleration"][0], 2) + pow(
                        elem["acceleration"][1], 2) + pow(elem["acceleration"][2], 2))/3)

                    if prequaternion == None:
                        train_unit.append(0)
                    else:
                        pdot = prequaternion[0]*elem["quaternion"][0] + prequaternion[1]*elem["quaternion"][1] + \
                            prequaternion[2]*elem["quaternion"][2] + \
                            prequaternion[3]*elem["quaternion"][3]
                        train_unit.append(math.degrees(
                            math.acos(round(pdot, 6))))
                    prequaternion = elem["quaternion"]

                train_x.append(train_unit)

        scaler = StandardScaler()
        scaler.fit(train_x)

        train_x_fit = scaler.transform(train_x)
        test_x_fit = scaler.transform(test_x)
        # mlp all in one
        # mlp = MLPRegressor(hidden_layer_sizes=(600,300,50), activation="tanh", solver="lbfgs", max_iter=800, learning_rate="constant", learning_rate_init =0.01)
        # mlp.fit(train_x_fit, train_y)
        # test_t_pred = mlp.predict(test_x_fit)
        # print("predict data", groupid, test_t_pred, test_y, [abs(test_t_pred[0][0]-test_y[0][0]), abs(test_t_pred[0][1]-test_y[0][1]), abs(test_t_pred[0][2]-test_y[0][2])])

        # # mlp single
        # mlp_pt = MLPRegressor(hidden_layer_sizes=(
        #     50, 50, 50, 50, 50), activation="tanh", solver="lbfgs")
        # mlp_pt.fit(train_x_fit, train_y_pt)
        # test_y_pt = mlp_pt.predict(test_x_fit)

        # mlp_sbp = MLPRegressor(hidden_layer_sizes=(
        #     50, 50, 50, 50, 50), activation="tanh", solver="lbfgs")
        # mlp_sbp.fit(train_x_fit, train_y_sbp)
        # test_y_sbp = mlp_sbp.predict(test_x_fit)

        # mlp_dbp = MLPRegressor(hidden_layer_sizes=(
        #     50, 50, 50, 50, 50), activation="tanh", solver="lbfgs")
        # mlp_dbp.fit(train_x_fit, train_y_dbp)
        # test_y_dbp = mlp_dbp.predict(test_x_fit)

        # print("mlp single", groupid, [
        #       test_y_pt[0], test_y_sbp[0], test_y_dbp[0]], [abs(test_y_pt[0]-test_y[0][0]), abs(test_y_sbp[0]-test_y[0][1]), abs(test_y_dbp[0]-test_y[0][2])])

        # mlperrors[0].append(round(abs(test_y_pt[0]-test_y[0][0]), 2))
        # mlperrors[1].append(round(abs(test_y_sbp[0]-test_y[0][1]), 2))
        # mlperrors[2].append(round(abs(test_y_dbp[0]-test_y[0][2]), 2))
        # mlperrors[3].append(groupid)

        # Bayesian Ridge Regression
        bayes_pt = linear_model.LinearRegression()
        bayes_pt.fit(train_x_fit, train_y_pt)
        test_y_pt = bayes_pt.predict(test_x_fit)

        bayes_sbp = linear_model.LinearRegression()
        bayes_sbp.fit(train_x_fit, train_y_sbp)
        test_y_sbp = bayes_sbp.predict(test_x_fit)

        bayes_dbp = linear_model.LinearRegression()
        bayes_dbp.fit(train_x_fit, train_y_dbp)
        test_y_dbp = bayes_dbp.predict(test_x_fit)

        print("linear", groupid, [
              test_y_pt[0], test_y_sbp[0], test_y_dbp[0]], [abs(test_y_pt[0]-test_y[0][0]), abs(test_y_sbp[0]-test_y[0][1]), abs(test_y_dbp[0]-test_y[0][2])])

        bayesianerrors[0].append(round(abs(test_y_pt[0]-test_y[0][0]), 2))
        bayesianerrors[1].append(round(abs(test_y_sbp[0]-test_y[0][1]), 2))
        bayesianerrors[2].append(round(abs(test_y_dbp[0]-test_y[0][2]), 2))
        bayesianerrors[3].append(groupid)

        res[0].append(test_y[0][0] - test_y_pt[0])
        res[1].append(test_y[0][1] - test_y_sbp[0])
        res[2].append(test_y[0][2] - test_y_dbp[0])
        res[3].append(groupid)

        bayesianpred[0].append(test_y_pt[0])
        bayesianpred[1].append(test_y_sbp[0])
        bayesianpred[2].append(test_y_dbp[0])
        bayesianpred[3].append(groupid)

        # # AdaBoost
        # adaboost_pt = AdaBoostRegressor(
        #     DecisionTreeRegressor(max_depth=6), n_estimators=900)
        # adaboost_pt.fit(train_x_fit, train_y_pt)
        # test_y_pt = adaboost_pt.predict(test_x_fit)

        # adaboost_sbp = AdaBoostRegressor(
        #     DecisionTreeRegressor(max_depth=6), n_estimators=900)
        # adaboost_sbp.fit(train_x_fit, train_y_sbp)
        # test_y_sbp = adaboost_sbp.predict(test_x_fit)

        # adaboost_dbp = AdaBoostRegressor(
        #     DecisionTreeRegressor(max_depth=6), n_estimators=900)
        # adaboost_dbp.fit(train_x_fit, train_y_dbp)
        # test_y_dbp = adaboost_dbp.predict(test_x_fit)

        # print("adaboost", groupid, [
        #       test_y_pt[0], test_y_sbp[0], test_y_dbp[0]], [abs(test_y_pt[0]-test_y[0][0]), abs(test_y_sbp[0]-test_y[0][1]), abs(test_y_dbp[0]-test_y[0][2])])

        # adaboosterrors[0].append(round(abs(test_y_pt[0]-test_y[0][0]), 2))
        # adaboosterrors[1].append(round(abs(test_y_sbp[0]-test_y[0][1]), 2))
        # adaboosterrors[2].append(round(abs(test_y_dbp[0]-test_y[0][2]), 2))
        # adaboosterrors[3].append(groupid)

        # # Nearest Neighbors
        # knn_pt = neighbors.KNeighborsRegressor(
        #     n_neighbors=5, weights="uniform")
        # knn_pt.fit(train_x_fit, train_y_pt)
        # test_y_pt = knn_pt.predict(test_x_fit)

        # knn_sbp = neighbors.KNeighborsRegressor(
        #     n_neighbors=5, weights="uniform")
        # knn_sbp.fit(train_x_fit, train_y_sbp)
        # test_y_sbp = knn_sbp.predict(test_x_fit)

        # knn_dbp = neighbors.KNeighborsRegressor(
        #     n_neighbors=5, weights="uniform")
        # knn_dbp.fit(train_x_fit, train_y_dbp)
        # test_y_dbp = knn_dbp.predict(test_x_fit)

        # print("knn", groupid, [
        #       test_y_pt[0], test_y_sbp[0], test_y_dbp[0]], [abs(test_y_pt[0]-test_y[0][0]), abs(test_y_sbp[0]-test_y[0][1]), abs(test_y_dbp[0]-test_y[0][2])])

        # knnerrors[0].append(round(abs(test_y_pt[0]-test_y[0][0]), 2))
        # knnerrors[1].append(round(abs(test_y_sbp[0]-test_y[0][1]), 2))
        # knnerrors[2].append(round(abs(test_y_dbp[0]-test_y[0][2]), 2))
        # knnerrors[3].append(groupid)

        # # Gradient Boosting
        # grad_pt = GradientBoostingRegressor(
        #     loss="huber", learning_rate=0.01, n_estimators=500, max_depth=8)
        # grad_pt.fit(train_x_fit, train_y_pt)
        # test_y_pt = grad_pt.predict(test_x_fit)

        # grad_sbp = GradientBoostingRegressor(
        #     loss="huber", learning_rate=0.01, n_estimators=500, max_depth=8)
        # grad_sbp.fit(train_x_fit, train_y_sbp)
        # test_y_sbp = grad_sbp.predict(test_x_fit)

        # grad_dbp = GradientBoostingRegressor(
        #     loss="huber", learning_rate=0.01, n_estimators=500, max_depth=8)
        # grad_dbp.fit(train_x_fit, train_y_dbp)
        # test_y_dbp = grad_dbp.predict(test_x_fit)

        # print("grad", groupid, [
        #       test_y_pt[0], test_y_sbp[0], test_y_dbp[0]], [abs(test_y_pt[0]-test_y[0][0]), abs(test_y_sbp[0]-test_y[0][1]), abs(test_y_dbp[0]-test_y[0][2])])

        # gradienterrors[0].append(round(abs(test_y_pt[0]-test_y[0][0]), 2))
        # gradienterrors[1].append(round(abs(test_y_sbp[0]-test_y[0][1]), 2))
        # gradienterrors[2].append(round(abs(test_y_dbp[0]-test_y[0][2]), 2))
        # gradienterrors[3].append(groupid)

        # # SVM
        # # svr_pt = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        # svr_pt = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3],
        #                                                                       "gamma": np.logspace(-2, 2, 5)})
        # svr_pt.fit(train_x_fit, train_y_pt)
        # test_y_pt = svr_pt.predict(test_x_fit)

        # # svr_sbp = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        # svr_sbp = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3],
        #                                                                        "gamma": np.logspace(-2, 2, 5)})
        # svr_sbp.fit(train_x_fit, train_y_sbp)
        # test_y_sbp = svr_sbp.predict(test_x_fit)

        # # svr_dbp = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        # svr_dbp = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3],
        #                                                                        "gamma": np.logspace(-2, 2, 5)})
        # svr_dbp.fit(train_x_fit, train_y_dbp)
        # test_y_dbp = svr_dbp.predict(test_x_fit)

        # print("svm", groupid, [test_y_pt[0], test_y_sbp[0], test_y_dbp[0]], [abs(
        #     test_y_pt[0]-test_y[0][0]), abs(test_y_sbp[0]-test_y[0][1]), abs(test_y_dbp[0]-test_y[0][2])])

        # svmerrors[0].append(round(abs(test_y_pt[0]-test_y[0][0]), 2))
        # svmerrors[1].append(round(abs(test_y_sbp[0]-test_y[0][1]), 2))
        # svmerrors[2].append(round(abs(test_y_dbp[0]-test_y[0][2]), 2))
        # svmerrors[3].append(groupid)

        pass

    for testindex in range(len(datalist)):

        groupid = datalist[testindex]["group"]
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        original_y = []
        pre_y = []
        

        # for single train
        train_y_pt = []
        train_y_sbp = []
        train_y_dbp = []

        # for single train mlp
        # train_y_pt_list = []
        # train_y_sbp_list = []
        # train_y_dbp_list = []

        # predict errors
        # predicterrors[0] = []
        # predicterrors[1] = []
        # predicterrors[2] = []

        for index in range(len(datalist)):
            group = datalist[index]
            if index == testindex:
                pre_y.append([bayesianpred[0][testindex], bayesianpred[1][testindex], bayesianpred[2][testindex]])
                test_y.append([res[0][testindex], res[1][testindex], res[2][testindex]])
                original_y.append([group["cuff"][1]["pulserate"], group["cuff"][1]
                               ["systolicbloodpressure"], group["cuff"][1]["diastolicbloodpressure"]])
                # test_x.append([])
                test_unit = []
                prequaternion = None
                for elem in group["sensor"][-60:]:
                    # print(elem["pulserate"], elem["systolicbloodpressure"], elem["diastolicbloodpressure"])
                    # print(elem["euler"], elem["quaternion"], elem["acceleration"])

                    test_unit.extend(
                        [elem["pulserate"], elem["systolicbloodpressure"], elem["diastolicbloodpressure"]])

                    test_unit.append(math.sqrt(pow(elem["acceleration"][0], 2) + pow(
                        elem["acceleration"][1], 2) + pow(elem["acceleration"][2], 2))/3)

                    if prequaternion == None:
                        test_unit.append(0)
                    else:
                        pdot = prequaternion[0]*elem["quaternion"][0] + prequaternion[1]*elem["quaternion"][1] + \
                            prequaternion[2]*elem["quaternion"][2] + \
                            prequaternion[3]*elem["quaternion"][3]
                        test_unit.append(math.degrees(
                            math.acos(round(pdot, 6))))
                    prequaternion = elem["quaternion"]

                test_x.append(test_unit)

            else:
                train_y.append([group["cuff"][1]["pulserate"], group["cuff"][1]
                                ["systolicbloodpressure"], group["cuff"][1]["diastolicbloodpressure"]])
                # train_x.append([])
                train_y_pt.append(res[0][index])
                train_y_sbp.append(res[1][index])
                train_y_dbp.append(res[2][index])

                # train_y_pt_list.append([group["cuff"][1]["pulserate"]])
                # train_y_sbp_list.append(
                #     [group["cuff"][1]["systolicbloodpressure"]])
                # train_y_dbp_list.append(
                #     [group["cuff"][1]["diastolicbloodpressure"]])

                train_unit = []
                prequaternion = None
                for elem in group["sensor"][-60:]:
                    # print(elem["pulserate"], elem["systolicbloodpressure"], elem["diastolicbloodpressure"])
                    # print(elem["euler"], elem["quaternion"], elem["acceleration"])

                    train_unit.extend(
                        [elem["pulserate"], elem["systolicbloodpressure"], elem["diastolicbloodpressure"]])

                    train_unit.append(math.sqrt(pow(elem["acceleration"][0], 2) + pow(
                        elem["acceleration"][1], 2) + pow(elem["acceleration"][2], 2))/3)

                    if prequaternion == None:
                        train_unit.append(0)
                    else:
                        pdot = prequaternion[0]*elem["quaternion"][0] + prequaternion[1]*elem["quaternion"][1] + \
                            prequaternion[2]*elem["quaternion"][2] + \
                            prequaternion[3]*elem["quaternion"][3]
                        train_unit.append(math.degrees(
                            math.acos(round(pdot, 6))))
                    prequaternion = elem["quaternion"]

                train_x.append(train_unit)

        scaler = StandardScaler()
        scaler.fit(train_x)

        train_x_fit = scaler.transform(train_x)
        test_x_fit = scaler.transform(test_x)

        # AdaBoost
        adaboost_pt = AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=6), n_estimators=900)
        adaboost_pt.fit(train_x_fit, train_y_pt)
        test_y_pt = adaboost_pt.predict(test_x_fit)

        adaboost_sbp = AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=6), n_estimators=900)
        adaboost_sbp.fit(train_x_fit, train_y_sbp)
        test_y_sbp = adaboost_sbp.predict(test_x_fit)

        adaboost_dbp = AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=6), n_estimators=900)
        adaboost_dbp.fit(train_x_fit, train_y_dbp)
        test_y_dbp = adaboost_dbp.predict(test_x_fit)

        print("adaboost", groupid, [
              test_y_pt[0], test_y_sbp[0], test_y_dbp[0]])

        adaboosterrors[0].append(round(abs(test_y_pt[0]+pre_y[0][0] - original_y[0][0]), 2))
        adaboosterrors[1].append(round(abs(test_y_sbp[0]+pre_y[0][1] - original_y[0][1]), 2))
        adaboosterrors[2].append(round(abs(test_y_dbp[0]+pre_y[0][2] - original_y[0][2]), 2))
        adaboosterrors[3].append(groupid)

    # mlperrors[0].reverse()
    # mlperrors[1].reverse()
    # mlperrors[2].reverse()
    # mlperrors[3].reverse()

    bayesianerrors[0].reverse()
    bayesianerrors[1].reverse()
    bayesianerrors[2].reverse()
    bayesianerrors[3].reverse()

    adaboosterrors[0].reverse()
    adaboosterrors[1].reverse()
    adaboosterrors[2].reverse()
    adaboosterrors[3].reverse()

    # knnerrors[0].reverse()
    # knnerrors[1].reverse()
    # knnerrors[2].reverse()
    # knnerrors[3].reverse()

    # gradienterrors[0].reverse()
    # gradienterrors[1].reverse()
    # gradienterrors[2].reverse()
    # gradienterrors[3].reverse()

    # svmerrors[0].reverse()
    # svmerrors[1].reverse()
    # svmerrors[2].reverse()
    # svmerrors[3].reverse()

    # # for error bar
    # fig = plt.figure(num="gradient-error", figsize=[10, 7])
    # # ids = [str(x) for x in range(len(datalist))]

    # x = list(range(len(datalist)))
    # total_width, n = 0.9, 3
    # width = total_width / n
    # # x = x - (total_width - width) / 2

    # errplot = fig.add_subplot(111)
    # errplot.bar(x, predicterrors[0], width=width, label="pulserate")

    # errplot.bar([c+width for c in x], predicterrors[1], width=width, label="systolicbp", tick_label=predicterrors[3])

    # errplot.bar([c+width*2 for c in x], predicterrors[2], width=width, label="diastolicbp")
    # errplot.legend()

    # errplot.set_xlabel("$group.id$")
    # errplot.set_title("gradient error")

    # plt.tight_layout()
    # plt.subplots_adjust(hspace=0.3)
    # plt.savefig("image/gradienterror.jpg")
    # # plt.show()

    return bayesianerrors, adaboosterrors


def show_err(ids, errors):
    # ids = ["original", "mlp", "bayesian", "adaboost", "knn", "gradient"]
    # errors = [originalerrors, mlperrors, bayesianerrors, adaboosterrors, knnerrors, gradienterrors]
    ptsum = []
    sbpsum = []
    dbpsum = []

    for elem in errors:
        ptsum.append(math.sqrt(sum([c*c for c in elem[0]])/len(elem[0])))
        sbpsum.append(math.sqrt(sum([c*c for c in elem[1]])/len(elem[1])))
        dbpsum.append(math.sqrt(sum([c*c for c in elem[2]])/len(elem[2])))

     # for error bar
    fig = plt.figure(num="error-compare", figsize=[10, 7])
    # ids = [str(x) for x in range(len(datalist))]

    x = list(range(len(errors)))
    total_width, n = 0.9, 3
    width = total_width / n
    # x = x - (total_width - width) / 2

    errplot = fig.add_subplot(111)
    errplot.bar(x, ptsum, width=width, label="pulserate")

    errplot.bar([c+width for c in x], sbpsum, width=width,
                label="systolicbp", tick_label=ids)

    errplot.bar([c+width*2 for c in x], dbpsum,
                width=width, label="diastolicbp")
    errplot.legend()

    errplot.set_xlabel("$group.id$")
    errplot.set_title("RMSD")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig("image/errorcompare2.jpg")
    # plt.show()

    pass


def main():

    datalist = read_data(cuffpath="data/cuff.csv",
                         personpath="data/person.csv", sensorpath="data/sensor.csv")

    originalerrors = show_data(datalist=datalist)

    bayesianerrors, adaboosterrors = train_data(datalist=datalist)

    # # show_err(originalerrors=originalerrors, mlperrors=mlperrors, bayesianerrors=bayesianerrors, adaboosterrors=adaboosterrors, knnerrors=knnerrors, gradienterrors=gradienterrors)

    show_err(ids=["original",  "linear", "adaboost"], errors=[
             originalerrors, bayesianerrors, adaboosterrors])


if __name__ == "__main__":
    main()
