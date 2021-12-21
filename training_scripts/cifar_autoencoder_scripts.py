import torch

device = (
    torch.device('cuda') if torch.cuda.is_available() 
    else torch.device('cpu')
)

def train_autoencoder_ensemble(
    ensemble,
    optimizer,
    scheduler,
    scheduler_epochs,
    encodings_criterion,
    decodings_criterion,
    anchor_criterion,
    train_dataloader,
    test_dataloader,
    num_epochs,
    show_print=True
):

    results = []

    for epoch in range(num_epochs):

        epoch_enc_loss, epoch_dec_loss, epoch_anc_loss, batch_count = 0, 0, 0, 0
        ensemble.train()

        for inputs, targets in train_dataloader:

            if inputs.shape[0] == 1:
                # Batch norm
                continue

            optimizer.zero_grad()

            inputs = inputs.to(device)

            encodings, decodings = ensemble(inputs)

            encodings_loss = encodings_criterion(encodings, inputs) if encodings_criterion else torch.tensor(0)
            decodings_loss = decodings_criterion(decodings, inputs) if decodings_criterion else torch.tensor(0)
            anchor_loss = anchor_criterion(encodings) if anchor_criterion else torch.tensor(0)
	          
            (encodings_loss + decodings_loss + anchor_loss).backward()
            optimizer.step()            

            epoch_enc_loss += encodings_loss.item()
            epoch_dec_loss += decodings_loss.item()
            epoch_anc_loss += anchor_loss.item()
            batch_count += 1

        test_epoch_enc_loss, test_epoch_dec_loss, test_epoch_anc_loss, test_batch_count = 0, 0, 0, 0
        ensemble.eval()

        for inputs, targets in test_dataloader:

            inputs = inputs.to(device)
            
            encodings, decodings = ensemble(inputs)

            encodings_loss = encodings_criterion(encodings, inputs) if encodings_criterion else torch.tensor(0)
            decodings_loss = decodings_criterion(decodings, inputs) if decodings_criterion else torch.tensor(0)
            anchor_loss = anchor_criterion(encodings) if anchor_criterion else torch.tensor(0)
	        
            test_epoch_enc_loss += encodings_loss.item()
            test_epoch_dec_loss += decodings_loss.item()
            test_epoch_anc_loss += anchor_loss.item()
            test_batch_count += 1

        if test_batch_count == 0:
            test_batch_count += 1
        if batch_count == 0:
            batch_count += 1

        if show_print:
            print_string = f'\n\nEpoch {epoch}'
            print_string += f'\n\ttrain_enc_loss {epoch_enc_loss/batch_count}'
            print_string += f'\n\ttrain_dec_loss {epoch_dec_loss/batch_count}'
            print_string += f'\n\ttrain_anc_loss {epoch_anc_loss/batch_count}'
            print_string += f'\n\ttest_enc_loss {test_epoch_enc_loss/test_batch_count}'
            print_string += f'\n\ttest_dec_loss {test_epoch_dec_loss/test_batch_count}'
            print_string += f'\n\ttest_anc_loss {test_epoch_anc_loss/test_batch_count}'
            print(print_string,flush=True,)
            
        if epoch + 1 in scheduler_epochs:
            scheduler.step()

        results.append(
            {
                'epoch': epoch,
                'train enc loss': float(epoch_enc_loss/batch_count), 
                'train dec loss': float(epoch_dec_loss/batch_count), 
                'train anc loss': float(epoch_anc_loss/batch_count),
                'test enc loss': float(test_epoch_enc_loss/test_batch_count), 
                'test dec loss': float(test_epoch_dec_loss/test_batch_count), 
                'test anc loss': float(test_epoch_anc_loss/test_batch_count)
            }
        )
    
    return ensemble, results

