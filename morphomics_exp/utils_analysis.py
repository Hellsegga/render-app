import numpy as np
import torch as th
import collections.abc

def deep_update(original, updates):
    for key, value in updates.items():
        if isinstance(value, collections.abc.Mapping) and key in original:
            deep_update(original[key], value)  # Recursively update nested dict
        else:
            original[key] = value  # Otherwise, overwrite/update key

def get_2d(pi):
    size = int(np.sqrt(pi.shape[0]))
    pi_2d = pi.reshape(size, size)
    return pi_2d

def mask_pi(pi, pixes_tokeep):    
    # Create a mask of zeros
    pi_threshold = np.zeros_like(pi)

    # Assign original values only at the specified indices
    pi_filtered = pi[pixes_tokeep]
    pi_threshold[pixes_tokeep] = pi_filtered
    return pi_threshold, pi_filtered

def get_base(pi, pixes_tokeep):
    # TODO modify 10000
    pi_full = np.zeros((10000))
    pi_full[pixes_tokeep] = pi
    return pi_full

def inverse_function(point, model, pca, scaler, filter):
    model.eval()
    with th.no_grad():
        # Pass the data through the model
        point_tensor = th.tensor(point, dtype=th.float32)
        out = model.decoder(point_tensor)
        pred_processed_pi = out.cpu().detach().numpy().reshape(1, -1)
        pred_scaled_pi = pca.inverse_transform(pred_processed_pi)
        pred_filter_pi = scaler.inverse_transform(pred_scaled_pi)
        if filter is None:
            return pred_filter_pi.flatten()  # Return the full vector without filtering
        pi = get_base(pred_filter_pi, filter)
        return pi 

def inverse_cnn_function(point, model):
    model.eval()
    with th.no_grad():
        # Pass the data through the model
        point_tensor = th.tensor(point, dtype=th.float32)
        point_tensor = point_tensor.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, 50, 50)

        out = model.decoder(point_tensor)
        normalized_pi = out.cpu().detach().numpy().squeeze()
        return normalized_pi 