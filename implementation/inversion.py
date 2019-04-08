import numpy as np

def invert(regressor, expected_output, input_size, step_size, iteration_count = 100, verbose = False):
    guessedInput = np.random.rand(input_size)
    layer_units = [[input_size] + list(regressor.hidden_layer_sizes) + [regressor.n_outputs_]]

    for j in range(0, iteration_count):
        activations = [guessedInput]

        for i in range(regressor.n_layers_ - 1):
            activations.append(np.empty((guessedInput.shape[0], layer_units[0][i + 1])))

        regressor._forward_pass(activations)
        y_pred = activations[-1]
        deltas = activations.copy()
        deltas[-1] = _activationFunctionDerivate(activations[-1], regressor.activation) * (y_pred - expected_output)

        for i in range(1, len(activations)):
            deltas[-i - 1] = _activationFunctionDerivate(activations[-i - 1], regressor.activation) * \
                             (regressor.coefs_[-i] * deltas[-i].T).sum(axis=1)
            if verbose:
                print('#', i)
                print(regressor.coefs_[-i])
                print(deltas[-i])
                print(regressor.coefs_[-i] * deltas[-i].T)
                print((regressor.coefs_[-i] * deltas[-i].T).sum(axis=1))
                print(activations[-i-1])
                print(_activationFunctionDerivate(activations[-i-1], regressor.activation))
                print(deltas[-i-1])
                print('-------------------')

        guessedInput = guessedInput - step_size * deltas[0]

    return guessedInput


def _activationFunctionDerivate(X, activation):
    if activation == 'tanh':
        return 1.0 / (np.cosh(X)**2)
    if activation == 'logistic':
        log_sigmoid = 1.0 / (1.0 + np.exp(-1 * X))
        return log_sigmoid * (1.0 - log_sigmoid)
    if activation == 'relu':
        return [1.0 if np.any(X > 0.0) else 0.0]

