data = {
    "Ex1": {
        "beta": [4, 10, 6.4, 2.9, 3.2],
        "g_abs": [19.9, 14.1, 16.3, 23.3, 22.0],
        "b": [5.1, 7.3, 6.4, 4.6, 4.7],
        "g_rel":[15.0, 10.7, 12.4, 17.7, 16.7],
        "f":[],
    },
    "Ex2": {
        "x_abs": [16.9, 18.3, 21.0],  # "None" for unclear rows
        "beta": [10/3, 10/2.5, 10/1.5],
        "t/f": [3.75, 3.75, 3.75],
    },
    "Ex3": {
        "div_width": 60
    },
    "Ex4": {
        "A_1": [0.2],
        "A_2": [1.0],
        "B_1": [0.3],
        "B_2": [0.6],
    }
}



data["Ex1"["f"]] = (data["Ex1"["g_rel"]] - data["Ex1"["b"]])


