import numpy as np
import pylab
import matplotlib.pyplot as plt

def dec_to_binarray(dec_num, len_array):
    function_result = np.zeros(len_array, dtype=int)
    binary_reprsent = np.binary_repr(dec_num, width=len_array)
    for i in range(len_array):
        function_result[i] = binary_reprsent[i]

    return function_result

# Parameter setting
len_message = 4
len_codeword = 7
num_message = 100000
transmitted_bit_stream = np.random.randint(2, size=(num_message,len_message))

# Generator matrix (7,4 Hamming code)
gen_matrix = [[1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
    ]

parity_check_matrix = [[1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1]]

# Encoding
encoded_bit_stream = np.matmul(transmitted_bit_stream, gen_matrix) % 2

# BPSK modulation
modulated_bit_stream = (-2 * encoded_bit_stream + 1) * np.sqrt(4/7)

# Additive white Gaussian noise
SNR_dBs = np.arange(0, 11, 1)
BER_hard_decoding_store = np.zeros(len(SNR_dBs))
BER_soft_decoding_store = np.zeros(len(SNR_dBs))
for iter, SNR_dB in enumerate(SNR_dBs):
    SNR_linear = 10 ** (SNR_dB/10)
    noise_variance = 0.5 / SNR_linear
    noise_stream = np.sqrt(noise_variance) * np.random.randn(num_message, len_codeword)

    # Received bit stream
    received_bit_stream = modulated_bit_stream + noise_stream

    # Demodulation of received bit stream
    demodulated_bit_stream = np.zeros((num_message,len_codeword), dtype = int)
    for i in range(num_message):
        for j in range(len_codeword):
            if received_bit_stream[i][j] < 0:
                demodulated_bit_stream[i][j] = 1

    # Hard decoding
    error_location = np.ones(num_message) * -1
    hard_decoded_bit_stream = demodulated_bit_stream
    transpose_parity_check_matrix = np.transpose(parity_check_matrix)
    syndrome_result = np.matmul(demodulated_bit_stream, transpose_parity_check_matrix) % 2
    for i in range(num_message):
        for j in range(len_codeword):
            if np.all(syndrome_result[i] == transpose_parity_check_matrix[j]):
                error_location[i] = j
                hard_decoded_bit_stream[i][j] = hard_decoded_bit_stream[i][j] + 1 % 2
                break

    # BER of hard decoding
    error_count_hard_decoding = 0
    for i in range(num_message):
        for j in range(len_message):
            if hard_decoded_bit_stream[i][j] != encoded_bit_stream[i][j]:
                error_count_hard_decoding += 1

    BER_hard_decoding_store[iter] = error_count_hard_decoding / (num_message*len_message)

    # Soft decoding
    codeword_table = np.zeros((2**len_message, len_codeword))
    for i in range(2**len_message):
        message_tmp = dec_to_binarray(i, len_message)
        codeword_table[i] = np.matmul(message_tmp, gen_matrix) % 2

    codeword_table_BPSK = -2 * codeword_table + 1
    correlation_value = np.matmul(received_bit_stream, np.transpose(codeword_table_BPSK))

    soft_decoded_bit_stream = np.zeros((num_message, len_codeword))
    for i in range(num_message):
        max_argument = np.argmax(correlation_value[i])
        soft_decoded_bit_stream[i] = codeword_table[max_argument]

    # BER of soft decoding
    error_count_soft_decoding = 0
    for i in range(num_message):
        for j in range(len_message):
            if soft_decoded_bit_stream[i][j] != encoded_bit_stream[i][j]:
                error_count_soft_decoding += 1

    BER_soft_decoding_store[iter] = error_count_soft_decoding / (num_message*len_message)


fig = plt.figure()
hard_decoding = fig.add_subplot(2, 1, 1)
line_hard, = hard_decoding.plot(SNR_dBs, BER_hard_decoding_store, color = 'red', lw=2)
soft_decoding = fig.add_subplot(2, 1, 2)
line_soft, = soft_decoding.plot(SNR_dBs, BER_soft_decoding_store, color = 'blue', lw=2)

hard_decoding.set_yscale('log')
soft_decoding.set_yscale('log')
plt.grid()
pylab.show()