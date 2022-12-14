"""variable node function for belief-propagation"""

from belief_propagation.edge_message import EdgeMessage
from belief_propagation.utils import coef_two_gaussian_multiplication, \
    mean_two_gaussian_multiplication, \
    variance_two_gaussian_multiplication

MIXTURE_COEEFICIENT = 1 / 3


def variable_node_function(channel_message, sigma, weights, means, variances):
    """
    compute the message from variable nodes

    :param channel_message: channel message, scalar
    :param sigma: the variance of channel noise
    :param weights: edge weights h1, h2, ..., hd-1 of check matrix
    :param means: means of message from check nodes for edge 1,2,...,d-1
    :param variances: variance of message from check nodes for edge 1,2,...,d-1

    :return:  m_out, v_out for edge d
    """
    edge_num = len(means)
    three_gaussians = [EdgeMessage() for i in range(edge_num)]
    # use m_edge, v_edge to form 3-Gaussian r_e(z)
    set_three_gaussians(edge_num, means, three_gaussians, variances, weights, channel_message)
    # initialize recursion, these three lists represents a_z_old
    m_z_old = [0] + three_gaussians[0].means
    v_z_old = [0] + three_gaussians[0].vars
    c_z_old = [0] + three_gaussians[0].coef
    # recursion
    for edge in range(2, edge_num + 1):
        m_z_old, v_z_old, c_z_old = recursive_compute_a_z(c_z_old, edge, m_z_old, three_gaussians,
                                                          v_z_old)
    m_with_channel_message, v_with_channel_message, c_with_channel_message_normlized = \
        update_with_channel_message(c_z_old, edge_num, m_z_old, sigma, v_z_old, channel_message)
    # moment matching
    m_out = sum((m_with_channel_message[k] * c_with_channel_message_normlized[k] for k in
                 range(1, 3 ** edge_num + 1)))
    v_out = sum(
        (c_with_channel_message_normlized[k] * (
                m_with_channel_message[k] ** 2 + v_with_channel_message[k]) for k in
         range(1, 3 ** edge_num + 1))) - m_out ** 2
    return m_out, max(v_out, 0.001)


def update_with_channel_message(c_z_old, edge_num, m_z_old, sigma, v_z_old, channel_message):
    """
    update all Gaussians using the channel message

    :param c_z_old: list of coefficients for Gaussian to be multiplied by channel message
    :param edge_num: number of edges(d-1)
    :param m_z_old: list of means for Gaussian to be multiplied by channel message
    :param sigma: variance of channel message
    :param v_z_old: list of variances for Gaussian to be multiplied by channel message
    :param channel_message: mean of channel message

    :return: list of means, variances and coefficients after updated by channel messages
    """
    # multiply each Gaussian with channel message
    v_with_channel_message = [0]
    m_with_channel_message = [0]
    c_with_channel_message = [0]
    for k in range(1, 3 ** edge_num + 1):
        v_tmp = variance_two_gaussian_multiplication(sigma, v_z_old[k])
        v_with_channel_message.append(v_tmp)
        # m_new, m_old, v_k, v_new, v_old
        m_with_channel_message.append(
            mean_two_gaussian_multiplication(channel_message, m_z_old[k], v_tmp, sigma,
                                             v_z_old[k]))
        # c for channel message is 1
        c_with_channel_message.append(
            coef_two_gaussian_multiplication(1, c_z_old[k], channel_message, m_z_old[k], sigma,
                                             v_z_old[k]))
        # normalize with_channel_message
    sum_c_with_channel_message = sum(c_with_channel_message)
    c_with_channel_message_normlized = [i / sum_c_with_channel_message for i in
                                        c_with_channel_message]
    return m_with_channel_message, v_with_channel_message, c_with_channel_message_normlized


def recursive_compute_a_z(c_z_old, edge, m_z_old, three_gaussians, v_z_old):
    """
    recursively compute Ae(z) using Ae(z) = Ae-1(z) * R(z)

    :param c_z_old: list of Gaussian coefficients of Ae-1(z)
    :param edge: id of new edge
    :param m_z_old: list of Gaussian means of Ae-1(z)
    :param three_gaussians: three-Gaussians for all edges
    :param v_z_old: list of Gaussian variances of Ae-1(z)

    :return: list of means, variances and coefs for Gaussian updated by a new three-Gaussian
    """
    # initialize a_z_new -- could be represented by three lists
    v_temp = [0] * (3 ** edge + 1)
    m_temp = [0] * (3 ** edge + 1)
    c_temp = [0] * (3 ** edge + 1)
    # get a new r_z
    r_z_new = three_gaussians[edge - 1]
    v_z_new = r_z_new.vars
    m_z_new = r_z_new.means
    c_z_new = r_z_new.coef
    for j in range(1, 3 ** (edge - 1) + 1):
        v_old = v_z_old[j]
        m_old = m_z_old[j]
        c_old = c_z_old[j]
        for i in range(1, 4):
            v_new = v_z_new[i - 1]
            c_new = c_z_new[i - 1]
            m_new = m_z_new[i - 1]
            k = 3 * (j - 1) + i
            # compute new variance, mean and coef for a new Gaussian
            v_k = variance_two_gaussian_multiplication(v_new, v_old)
            m_k = mean_two_gaussian_multiplication(m_new, m_old, v_k, v_new, v_old)
            c_k = coef_two_gaussian_multiplication(c_new, c_old, m_new, m_old, v_new, v_old)
            # refresh the a_z_new
            v_temp[k] = v_k
            m_temp[k] = m_k
            c_temp[k] = c_k
    # reset r_z_old
    m_z_old = m_temp
    v_z_old = v_temp
    c_z_old = c_temp
    return m_z_old, v_z_old, c_z_old


def set_three_gaussians(edge_num, means, three_gaussians, variances, weights, channel_message):
    """
    compute the three-Gaussian for each incoming edge for a variable node

    :param edge_num: number of edges
    :param means: list of means for all incoming edges
    :param three_gaussians: list of three-Gaussians(initialized)
    :param variances: list of variances for all incoming edges
    :param weights: list of weights for all incoming edges, i.e. h for check matrix
    :param channel_message: mean of channel message

    :return: list of three-Gaussians
    """
    for i in range(edge_num):
        h_edge = weights[i]
        v_in = variances[i]
        m_in = means[i]
        b_edge = round(h_edge * (channel_message - m_in))
        three_gaussian_edge = three_gaussians[i]
        for j in range(1, 4):
            three_gaussian_edge.vars[j - 1] = v_in
            three_gaussian_edge.means[j - 1] = m_in + (b_edge + j - 2) / h_edge
            three_gaussian_edge.coef[j - 1] = MIXTURE_COEEFICIENT
        three_gaussians[i] = three_gaussian_edge
