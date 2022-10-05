"""Module for LDLC decoding using belief propagation"""
import numpy as np
from numpy.linalg import inv
from belief_propagation.variable_node import variable_node_function

VARIANCE_INDEX = 2
MEAN_INDEX = 1
WEIGHT_INDEX = 0


def ldlc_decoder(y, sigma, H, lmax):
    """
    LDLC Decoder

    :param y: received sequence, a vector
    :param sigma: channel variance
    :param H: m by n check matrix
    :param lmax: max iteration
    :return:
    """
    n = len(H[0])
    m = len(H)
    # initialize the message dict for each node
    # form like {variable_node: {check_node:[hji, mean, variance]}}
    variable_node_message_dict = {}
    # form like {check_node: {variable node:[hji, mean, variance]}}
    # represents a check node receive a Gaussian message from a variable node
    check_node_message_dict = {}
    # for each check node j
    for j in range(m):
        one_check = H[j]
        # for each variable node i
        for i in range(n):
            if one_check[i] != 0:
                if j not in check_node_message_dict:
                    check_node_message_dict[j] = {i: [one_check[i], y[i], sigma]}
                else:
                    check_node_message_dict[j].update({i: [one_check[i], y[i], sigma]})
                if i not in variable_node_message_dict:
                    variable_node_message_dict[i] = {j: [one_check[i], 0, 0]}
                else:
                    variable_node_message_dict[i].update({j: [one_check[i], 0, 0]})

    # iterative decoding
    for l in range(lmax):
        # for each check node
        for check_node in check_node_message_dict:
            variable_nodes_set = set(check_node_message_dict[check_node].keys())
            variable_info_for_one_check_onde = check_node_message_dict[check_node]
            for variable_node in variable_info_for_one_check_onde:
                edge_weight = variable_info_for_one_check_onde[variable_node][WEIGHT_INDEX]
                variable_nodes_set_exclude_one = variable_nodes_set - {variable_node}

                mean = -1 / edge_weight * sum(variable_info_for_one_check_onde[node][WEIGHT_INDEX] *
                                              variable_info_for_one_check_onde[node][MEAN_INDEX] for
                                              node in list(variable_nodes_set_exclude_one))
                variance = 1 / (edge_weight ** 2) * sum(
                    variable_info_for_one_check_onde[node][WEIGHT_INDEX] ** 2 *
                    variable_info_for_one_check_onde[node][VARIANCE_INDEX] for
                    node in list(variable_nodes_set_exclude_one))
                variable_node_message_dict[variable_node][check_node][MEAN_INDEX] = mean
                variable_node_message_dict[variable_node][check_node][VARIANCE_INDEX] = variance

        for variable_node in variable_node_message_dict:
            # channel_message, sigma, weights, means, variances
            check_nodes_set = set(variable_node_message_dict[variable_node].keys())
            check_node_info_for_one_variable_node = variable_node_message_dict[variable_node]
            for check_node in check_node_info_for_one_variable_node:
                set_exclude_one = check_nodes_set - {check_node}
                means_from_other_check_nodes = [
                    check_node_info_for_one_variable_node[node][MEAN_INDEX] for node in
                    list(set_exclude_one)]
                variances_from_other_check_nodes = [
                    check_node_info_for_one_variable_node[node][VARIANCE_INDEX] for node in
                    list(set_exclude_one)]
                weights_from_other_check_nodes = [
                    check_node_info_for_one_variable_node[node][WEIGHT_INDEX] for node in
                    list(set_exclude_one)]
                mean_to_this_check_node, variance_to_this_check_node = variable_node_function(
                    y[variable_node], sigma, weights_from_other_check_nodes,
                    means_from_other_check_nodes, variances_from_other_check_nodes)
                check_node_message_dict[check_node][variable_node][
                    MEAN_INDEX] = mean_to_this_check_node
                check_node_message_dict[check_node][variable_node][
                    VARIANCE_INDEX] = variance_to_this_check_node

    x_ = [0] * n
    for variable_node in variable_node_message_dict:
        check_node_info_for_one_variable_node = variable_node_message_dict[variable_node]
        means = [check_node_info_for_one_variable_node[node][MEAN_INDEX] for node in
                 check_node_info_for_one_variable_node]
        vars = [check_node_info_for_one_variable_node[node][VARIANCE_INDEX] for node in
                check_node_info_for_one_variable_node]
        edge_weights = [check_node_info_for_one_variable_node[node][WEIGHT_INDEX] for node in
                        check_node_info_for_one_variable_node]
        x_[variable_node], _ = variable_node_function(y[variable_node], sigma, edge_weights, means,
                                                     vars)

    b_hat = np.round(np.matmul(np.array(H), np.array(x_)))
    return np.round(np.matmul(inv(np.array(H)), b_hat))