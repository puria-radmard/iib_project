from config.ootb.creation_functions import make_recurrent_regression_architecture


def short_recurrent_regression_network(cell_type, dropout, device):
    return  make_recurrent_regression_architecture(
        cell_type, encoder_lstm_layers=[3,3], encoder_lstm_sizes=[256,128], 
        decoder_fc_hidden_dims=[128,16,1], embedding_dim=128, 
        dropout=dropout, device=device, feature_dim=40, ensemble_type='normal'
    )


def long_recurrent_regression_network(cell_type, dropout, device):
    return make_recurrent_regression_architecture(
        cell_type, encoder_lstm_layers=[3,3], encoder_lstm_sizes=[256,128], 
        decoder_fc_hidden_dims=[128,64,16,1], embedding_dim=128, 
        dropout=dropout, device=device, feature_dim=40, ensemble_type='normal'
    )
