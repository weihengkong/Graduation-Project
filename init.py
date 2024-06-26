node1 = {'id': 0, 'cpu': 4, 'mem': 16, 'gpu': 4, 'net': 100, 'price': 30.39, 'power': 100, 'place': []}
node2 = {'id': 1, 'cpu': 4, 'mem': 16, 'gpu': 8, 'net': 100, 'price': 60.94, 'power': 150, 'place': []}
node3 = {'id': 2, 'cpu': 8, 'mem': 32, 'gpu': 16, 'net': 100, 'price': 130.88, 'power': 200, 'place': []}
node4 = {'id': 3, 'cpu': 6, 'mem': 32, 'gpu': 24, 'net': 100, 'price': 150.09, 'power': 100, 'place': []}
node5 = {'id': 4, 'cpu': 8, 'mem': 40, 'gpu': 32, 'net': 100, 'price': 180.04, 'power': 170, 'place': []}
node6 = {'id': 5, 'cpu': 16, 'mem': 40, 'gpu': 32, 'net': 100, 'price': 190.10, 'power': 210, 'place': []}
nodes = [node1, node2, node3, node4, node5, node6]
#
# container1 = {'id': 0, 'cpu': 4, 'mem': 4, 'gpu': 2, 'net': 10, 'pre': [], 'after': [1, 2], 'front': []}
# container2 = {'id': 1, 'cpu': 2, 'mem': 4, 'gpu': 0, 'net': 15, 'pre': [0], 'after': [3], 'front': [0]}
# container3 = {'id': 2, 'cpu': 4, 'mem': 8, 'gpu': 4, 'net': 10, 'pre': [0], 'after': [3], 'front': [0]}
# container4 = {'id': 3, 'cpu': 4, 'mem': 8, 'gpu': 8, 'net': 15, 'pre': [0, 1, 2], 'after': [4], 'front': [1, 2]}
# container5 = {'id': 4, 'cpu': 4, 'mem': 8, 'gpu': 8, 'net': 10, 'pre': [0, 1, 2, 3], 'after': [], 'front': [3]}
# containers = [container1, container2, container3, container4, container5]
#
# trans_delay1 = [0, 0, 19, 0, 0]
# trans_delay2 = [0, 0, 24, 0, 0]
# trans_delay3 = [0, 0, 0, 21, 17]
# trans_delay4 = [0, 0, 0, 0, 0]
# trans_delay5 = [0, 0, 0, 0, 0]
# trans_delays = [trans_delay1, trans_delay2, trans_delay3, trans_delay4, trans_delay5]
# exc_delay1 = [22, 21, 18, 16, 15, 13]
# exc_delay2 = [23, 22, 17, 16, 14, 14]
# exc_delay3 = [22, 20, 17, 17, 15, 15]
# exc_delay4 = [23, 20, 18, 16, 14, 14]
# exc_delay5 = [22, 21, 17, 17, 15, 14]
# exc_delays = [exc_delay1, exc_delay2, exc_delay3, exc_delay4, exc_delay5]

# container1 = {'id': 0, 'cpu': 2, 'mem': 4, 'gpu': 2, 'net': 10, 'pre': [], 'after': [1, 2, 3], 'front': []}
# container2 = {'id': 1, 'cpu': 4, 'mem': 6, 'gpu': 4, 'net': 15, 'pre': [0], 'after': [4], 'front': [0]}
# container3 = {'id': 2, 'cpu': 4, 'mem': 8, 'gpu': 8, 'net': 10, 'pre': [0], 'after': [4], 'front': [0]}
# container4 = {'id': 3, 'cpu': 8, 'mem': 16, 'gpu': 8, 'net': 15, 'pre': [0], 'after': [4], 'front': [0]}
# container5 = {'id': 4, 'cpu': 4, 'mem': 4, 'gpu': 4, 'net': 10, 'pre': [0, 1, 2, 3], 'after': [5, 6, 7],
#               'front': [1, 2, 3]}
# container6 = {'id': 5, 'cpu': 2, 'mem': 2, 'gpu': 12, 'net': 15, 'pre': [0, 1, 2, 3, 4], 'after': [8], 'front': [4]}
# container7 = {'id': 6, 'cpu': 4, 'mem': 8, 'gpu': 6, 'net': 10, 'pre': [0, 1, 2, 3, 4], 'after': [8], 'front': [4]}
# container8 = {'id': 7, 'cpu': 8, 'mem': 8, 'gpu': 2, 'net': 15, 'pre': [0, 1, 2, 3, 4], 'after': [8], 'front': [4]}
# container9 = {'id': 8, 'cpu': 4, 'mem': 8, 'gpu': 4, 'net': 10, 'pre': [0, 1, 2, 3, 4, 5, 6, 7], 'after': [9],'front': [5,6,7]}
# container10 = {'id': 9, 'cpu': 8, 'mem': 16, 'gpu': 2, 'net': 15, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'after': [],'front': [8]}
# containers = [container1, container2, container3, container4, container5, container6, container7, container8,
#               container9, container10]
#
# trans_delay1 = [0, 13, 18, 21, 0, 0, 0, 0, 0, 0]
# trans_delay2 = [0, 0, 0, 0, 27, 0, 0, 0, 0, 0]
# trans_delay3 = [0, 0, 0, 0, 14, 0, 0, 0, 0, 0]
# trans_delay4 = [0, 0, 0, 0, 9, 0, 0, 0, 0, 0]
# trans_delay5 = [0, 0, 0, 0, 0, 29, 15, 16, 0, 0]
# trans_delay6 = [0, 0, 0, 0, 0, 0, 0, 0, 6, 0]
# trans_delay7 = [0, 0, 0, 0, 0, 0, 0, 0, 31, 0]
# trans_delay8 = [0, 0, 0, 0, 0, 0, 0, 0, 19, 0]
# trans_delay9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 34]
# trans_delay10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# trans_delays = [trans_delay1, trans_delay2, trans_delay3, trans_delay4, trans_delay5, trans_delay6, trans_delay7,
#                 trans_delay8, trans_delay9, trans_delay10]
#
# exc_delay1 = [22, 21, 18, 16, 15, 13]
# exc_delay2 = [23, 22, 17, 16, 14, 14]
# exc_delay3 = [22, 20, 17, 17, 15, 15]
# exc_delay4 = [23, 20, 18, 16, 14, 14]
# exc_delay5 = [22, 21, 17, 17, 15, 14]
# exc_delay6 = [23, 20, 18, 16, 16, 15]
# exc_delay7 = [22, 21, 17, 15, 15, 13]
# exc_delay8 = [23, 20, 17, 16, 14, 14]
# exc_delay9 = [22, 20, 18, 17, 15, 14]
# exc_delay10 = [23, 20, 18, 16, 15, 14]
# exc_delays = [exc_delay1, exc_delay2, exc_delay3, exc_delay4, exc_delay5, exc_delay6, exc_delay7, exc_delay8,
#               exc_delay9, exc_delay10]

# 20个
# container1 = {'id': 0, 'cpu': 4, 'mem': 4, 'gpu': 2, 'net': 10, 'pre': [], 'front': [], 'after': [4, 5]}
# container2 = {'id': 1, 'cpu': 3, 'mem': 3, 'gpu': 1, 'net': 15, 'pre': [], 'front': [], 'after': [6]}
# container3 = {'id': 2, 'cpu': 4, 'mem': 4, 'gpu': 0, 'net': 10, 'pre': [], 'front': [], 'after': [7]}
# container4 = {'id': 3, 'cpu': 4, 'mem': 2, 'gpu': 2, 'net': 15, 'pre': [], 'front': [], 'after': [8, 9]}
# container5 = {'id': 4, 'cpu': 3, 'mem': 3, 'gpu': 4, 'net': 10, 'pre': [0], 'front': [0], 'after': [10]}
# container6 = {'id': 5, 'cpu': 6, 'mem': 4, 'gpu': 2, 'net': 15, 'pre': [0], 'front': [0], 'after': [10]}
# container7 = {'id': 6, 'cpu': 4, 'mem': 3, 'gpu': 4, 'net': 10, 'pre': [1], 'front': [1], 'after': [10]}
# container8 = {'id': 7, 'cpu': 4, 'mem': 4, 'gpu': 2, 'net': 15, 'pre': [2], 'front': [2], 'after': [10]}
# container9 = {'id': 8, 'cpu': 8, 'mem': 2, 'gpu': 0, 'net': 10, 'pre': [3], 'front': [3], 'after': [10]}
# container10 = {'id': 9, 'cpu': 3, 'mem': 3, 'gpu': 2, 'net': 15, 'pre': [3], 'front': [3], 'after': [10]}
# container11 = {'id': 10, 'cpu': 4, 'mem': 4, 'gpu': 2, 'net': 10, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                'front': [4, 5, 6, 7, 8, 9], 'after': [11]}
# container12 = {'id': 11, 'cpu': 4, 'mem': 2, 'gpu': 4, 'net': 15, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                'front': [10], 'after': [12, 13, 14, 15]}
# container13 = {'id': 12, 'cpu': 8, 'mem': 3, 'gpu': 2, 'net': 10, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#                'front': [11], 'after': [16]}
# container14 = {'id': 13, 'cpu': 6, 'mem': 4, 'gpu': 1, 'net': 15, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#                'front': [11], 'after': [16]}
# container15 = {'id': 14, 'cpu': 6, 'mem': 2, 'gpu': 2, 'net': 10, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#                'front': [11], 'after': [17]}
# container16 = {'id': 15, 'cpu': 4, 'mem': 2, 'gpu': 2, 'net': 15, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#                'front': [11], 'after': [17]}
# container17 = {'id': 16, 'cpu': 4, 'mem': 2, 'gpu': 1, 'net': 10,
#                'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 'front': [12, 13], 'after': [18]}
# container18 = {'id': 17, 'cpu': 12, 'mem': 3, 'gpu': 1, 'net': 15,
#                'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15], 'front': [14, 15], 'after': [18]}
# container19 = {'id': 18, 'cpu': 3, 'mem': 4, 'gpu': 2, 'net': 10,
#                'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], 'front': [16, 17], 'after': [19]}
# container20 = {'id': 19, 'cpu': 4, 'mem': 2, 'gpu': 2, 'net': 15,
#                'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'front': [18], 'after': []}
# containers = [container1, container2, container3, container4, container5, container6, container7, container8,
#               container9, container10, container11, container12, container13, container14, container15, container16,
#               container17, container18,
#               container19, container20]
# trans_delays = [
#     [0, 0, 0, 0, 11, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 21, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 19, 11, 9, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ]
# exc_delays = [
#     [22, 21, 18, 16, 16, 14],
#     [23, 22, 17, 15, 15, 15],
#     [23, 21, 17, 15, 15, 13],
#     [24, 21, 18, 16, 15, 15],
#     [24, 20, 19, 17, 15, 13],
#     [23, 21, 18, 16, 16, 14],
#     [22, 22, 19, 17, 15, 13],
#     [24, 21, 18, 16, 15, 13],
#     [23, 20, 19, 17, 16, 14],
#     [22, 22, 18, 16, 15, 12],
#     [24, 21, 17, 16, 16, 14],
#     [22, 22, 18, 17, 16, 13],
#     [23, 21, 19, 17, 16, 14],
#     [22, 20, 18, 16, 15, 13],
#     [24, 22, 17, 16, 15, 12],
#     [23, 22, 17, 16, 15, 13],
#     [24, 22, 18, 16, 16, 14],
#     [23, 21, 19, 18, 15, 13],
#     [24, 22, 17, 16, 15, 13],
#     [23, 19, 17, 17, 17, 15],
# ]

# 30个

# container1 = {'id': 0, 'cpu': 2, 'mem': 4, 'gpu': 2, 'net': 10, 'pre': [], 'front': [], 'after': [4]}
# container2 = {'id': 1, 'cpu': 2, 'mem': 2, 'gpu': 0, 'net': 10, 'pre': [], 'front': [], 'after': [5]}
# container3 = {'id': 2, 'cpu': 2, 'mem': 2, 'gpu': 2, 'net': 10, 'pre': [], 'front': [], 'after': [6]}
# container4 = {'id': 3, 'cpu': 0.5, 'mem': 4, 'gpu': 2, 'net': 15, 'pre': [], 'front': [], 'after': [7]}
# container5 = {'id': 4, 'cpu': 2, 'mem': 4, 'gpu': 1, 'net': 10, 'pre': [0], 'front': [0], 'after': [8]}
# container6 = {'id': 5, 'cpu': 3, 'mem': 4, 'gpu': 0, 'net': 10, 'pre': [1], 'front': [1], 'after': [8]}
# container7 = {'id': 6, 'cpu': 1, 'mem': 3, 'gpu': 2, 'net': 10, 'pre': [2], 'front': [2], 'after': [8]}
# container8 = {'id': 7, 'cpu': 1, 'mem': 2, 'gpu': 0, 'net': 10, 'pre': [3], 'front': [3], 'after': [8]}
# container9 = {'id': 8, 'cpu': 2, 'mem': 4, 'gpu': 2, 'net': 10, 'pre': [0, 1, 2, 3, 4, 5, 6, 7],
#               'front': [4, 5, 6, 7], 'after': [9, 10, 11, 12]}
# container10 = {'id': 9, 'cpu': 2, 'mem': 5, 'gpu': 1, 'net': 10, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8],
#                'front': [8], 'after': [13]}
# container11 = {'id': 10, 'cpu': 2, 'mem': 2, 'gpu': 4, 'net': 15, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'front': [8],
#                'after': [13]}
# container12 = {'id': 11, 'cpu': 2, 'mem': 3, 'gpu': 2, 'net': 10, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'front': [8],
#                'after': [14]}
# container13 = {'id': 12, 'cpu': 2, 'mem': 2, 'gpu': 3, 'net': 15, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'front': [8],
#                'after': [14]}
# container14 = {'id': 13, 'cpu': 1, 'mem': 4, 'gpu': 2, 'net': 10, 'pre': [0, 1, 2, 3, 4, 5, 8, 9, 10],
#                'front': [9, 10], 'after': [29]}
# container15 = {'id': 14, 'cpu': 1, 'mem': 4, 'gpu': 3, 'net': 15, 'pre': [0, 1, 2, 3, 4, 5, 8, 11, 12],
#                'front': [11, 12], 'after': [29]}
# container16 = {'id': 15, 'cpu': 1, 'mem': 4, 'gpu': 4, 'net': 10, 'pre': [], 'front': [], 'after': [16, 17, 18]}
# container17 = {'id': 16, 'cpu': 0.5, 'mem': 3, 'gpu': 2, 'net': 10, 'pre': [15], 'front': [15], 'after': [19]}
# container18 = {'id': 17, 'cpu': 1, 'mem': 4, 'gpu': 4, 'net': 10, 'pre': [15], 'front': [15], 'after': [20]}
# container19 = {'id': 18, 'cpu': 0.5, 'mem': 3, 'gpu': 4, 'net': 10,
#                'pre': [15], 'front': [15], 'after': [21]}
# container20 = {'id': 19, 'cpu': 2, 'mem': 2, 'gpu': 2, 'net': 10,
#                'pre': [15, 16], 'front': [16], 'after': [22]}
# container21 = {'id': 20, 'cpu': 2, 'mem': 4, 'gpu': 2, 'net': 10, 'pre': [15, 17], 'front': [17], 'after': [22]}
# container22 = {'id': 21, 'cpu': 2, 'mem': 3, 'gpu': 3, 'net': 15, 'pre': [15, 18], 'front': [18], 'after': [22]}
# container23 = {'id': 22, 'cpu': 1, 'mem': 2, 'gpu': 4, 'net': 10, 'pre': [15, 16, 17, 18, 19, 20, 21],
#                'front': [19, 20, 21], 'after': [23, 24, 25, 26]}
# container24 = {'id': 23, 'cpu': 2, 'mem': 2, 'gpu': 2, 'net': 10, 'pre': [15, 16, 17, 18, 19, 20, 21, 22],
#                'front': [22], 'after': [27]}
# container25 = {'id': 24, 'cpu': 1, 'mem': 3, 'gpu': 2, 'net': 15, 'pre': [15, 16, 17, 18, 19, 20, 21, 22],
#                'front': [22], 'after': [27]}
# container26 = {'id': 25, 'cpu': 2, 'mem': 2, 'gpu': 3, 'net': 10, 'pre': [15, 16, 17, 18, 19, 20, 21, 22],
#                'front': [22], 'after': [28]}
# container27 = {'id': 26, 'cpu': 2, 'mem': 6, 'gpu': 2, 'net': 10, 'pre': [15, 16, 17, 18, 19, 20, 21, 22],
#                'front': [22], 'after': [27]}
# container28 = {'id': 27, 'cpu': 2, 'mem': 4, 'gpu': 4, 'net': 15, 'pre': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
#                'front': [23, 24], 'after': [29]}
# container29 = {'id': 28, 'cpu': 1, 'mem': 4, 'gpu': 2, 'net': 10,
#                'pre': [15, 16, 17, 18, 19, 20, 21, 22, 25, 26], 'front': [25, 26], 'after': [29]}
# container30 = {'id': 29, 'cpu': 2, 'mem': 4, 'gpu': 1, 'net': 10,
#                'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
#                        27, 28], 'front': [13, 14, 27, 28], 'after': []}
# containers = [container1, container2, container3, container4, container5, container6, container7, container8,
#               container9, container10, container11, container12, container13, container14, container15, container16,
#               container17, container18,
#               container19, container20, container21, container22, container23, container24, container25, container26,
#               container27, container28,
#               container29, container30]
#
# trans_delays = [
#     [0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 31, 23, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 9, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 31, 17, 26, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ]
# exc_delays = [
#     [22, 21, 18, 16, 16, 14],
#     [23, 22, 17, 15, 15, 15],
#     [23, 21, 17, 15, 15, 13],
#     [24, 21, 18, 16, 15, 15],
#     [24, 20, 19, 17, 15, 13],
#     [23, 21, 18, 16, 16, 14],
#     [22, 22, 19, 17, 15, 13],
#     [24, 21, 18, 16, 15, 13],
#     [23, 21, 19, 17, 16, 14],
#     [22, 22, 18, 16, 15, 13],
#     [24, 21, 17, 16, 16, 14],
#     [22, 22, 18, 17, 16, 13],
#     [23, 21, 19, 17, 16, 14],
#     [22, 20, 18, 16, 15, 13],
#     [24, 22, 17, 16, 15, 14],
#     [23, 22, 17, 16, 15, 13],
#     [24, 22, 18, 16, 16, 14],
#     [23, 21, 19, 18, 15, 13],
#     [24, 22, 17, 16, 15, 13],
#     [23, 20, 17, 17, 17, 15],
#     [24, 21, 17, 16, 16, 14],
#     [22, 22, 18, 17, 16, 13],
#     [23, 21, 19, 17, 16, 14],
#     [22, 20, 18, 16, 15, 13],
#     [24, 22, 17, 16, 15, 14],
#     [23, 22, 17, 16, 15, 13],
#     [24, 22, 18, 16, 16, 14],
#     [23, 21, 19, 18, 15, 13],
#     [24, 22, 17, 16, 15, 13],
#     [23, 20, 17, 17, 17, 15],
# ]


# 15个
container1 = {'id': 0, 'cpu': 1, 'mem': 4, 'gpu': 2, 'net': 10, 'pre': [], 'front': [], 'after': [4, 5]}
container2 = {'id': 1, 'cpu': 1, 'mem': 3, 'gpu': 1, 'net': 15, 'pre': [], 'front': [], 'after': [6]}
container3 = {'id': 2, 'cpu': 2, 'mem': 4, 'gpu': 0, 'net': 10, 'pre': [], 'front': [], 'after': [7]}
container4 = {'id': 3, 'cpu': 2, 'mem': 3, 'gpu': 2, 'net': 15, 'pre': [], 'front': [], 'after': [8, 9]}
container5 = {'id': 4, 'cpu': 2, 'mem': 3, 'gpu': 4, 'net': 10, 'pre': [0], 'front': [0], 'after': [10]}
container6 = {'id': 5, 'cpu': 3, 'mem': 4, 'gpu': 2, 'net': 15, 'pre': [0], 'front': [0], 'after': [10]}
container7 = {'id': 6, 'cpu': 2, 'mem': 3, 'gpu': 4, 'net': 10, 'pre': [1], 'front': [1], 'after': [10]}
container8 = {'id': 7, 'cpu': 4, 'mem': 4, 'gpu': 2, 'net': 15, 'pre': [2], 'front': [2], 'after': [10]}
container9 = {'id': 8, 'cpu': 6, 'mem': 1, 'gpu': 0, 'net': 10, 'pre': [3], 'front': [3], 'after': [10]}
container10 = {'id': 9, 'cpu': 4, 'mem': 3, 'gpu': 2, 'net': 15, 'pre': [3], 'front': [3], 'after': [10]}
container11 = {'id': 10, 'cpu': 2, 'mem': 4, 'gpu': 2, 'net': 10, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               'front': [4, 5, 6, 7, 8, 9], 'after': [11]}
container12 = {'id': 11, 'cpu': 2, 'mem': 4, 'gpu': 4, 'net': 15, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               'front': [10], 'after': [12, 13, 14]}
container13 = {'id': 12, 'cpu': 4, 'mem': 3, 'gpu': 2, 'net': 10, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
               'front': [11], 'after': []}
container14 = {'id': 13, 'cpu': 4, 'mem': 4, 'gpu': 1, 'net': 15, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
               'front': [11], 'after': []}
container15 = {'id': 14, 'cpu': 3, 'mem': 6, 'gpu': 2, 'net': 10, 'pre': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
               'front': [11], 'after': []}
containers = [container1, container2, container3, container4, container5, container6, container7, container8,
              container9, container10, container11, container12, container13, container14, container15]
trans_delays = [
    [0, 0, 0, 0, 11, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 21, 15, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 19, 11],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
# 每个node执行每个任务的时间
exc_delays = [
    [22, 21, 18, 16, 16, 14],
    [23, 22, 17, 15, 15, 15],
    [23, 21, 17, 15, 15, 13],
    [24, 21, 18, 16, 15, 15],
    [24, 20, 19, 17, 15, 13],
    [23, 21, 18, 16, 16, 14],
    [22, 22, 19, 17, 15, 13],
    [24, 21, 18, 16, 15, 13],
    [23, 20, 19, 17, 16, 14],
    [22, 22, 18, 16, 15, 12],
    [24, 21, 17, 16, 16, 14],
    [22, 22, 18, 17, 16, 13],
    [23, 21, 19, 17, 16, 14],
    [22, 20, 18, 16, 15, 13],
    [24, 22, 17, 16, 15, 12],
    [23, 22, 17, 16, 15, 13],
    [24, 22, 18, 16, 16, 14],
    [23, 21, 19, 18, 15, 13],
    [24, 22, 17, 16, 15, 13],
    [23, 19, 17, 17, 17, 15],
]

ranks = {}


# 根据前置容器的数量，对容器进行排序
def rank(container):
    if len(container['pre']) == 0:
        return 0
    else:
        pre_rank = []
        for pre_container_id in container['pre']:
            pre_rank.append(rank(containers[pre_container_id]))
        return max(pre_rank) + 1


for container in containers:
    r = rank(container)
    container['rank'] = r
    if 'rank' + str(r) in ranks.keys():
        ranks['rank' + str(r)].append(container['id'])
    else:
        ranks['rank' + str(r)] = [container['id']]
