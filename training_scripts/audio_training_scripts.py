import torch
from config import DEFAULT_DTYPE
device = (
    torch.device('cuda') if torch.cuda.is_available() 
    else torch.device('cpu')
)
from tqdm import tqdm


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
    show_print=True,
    dtype=DEFAULT_DTYPE
):

    results = []

    for epoch in range(num_epochs):

        epoch_enc_loss, epoch_dec_loss, epoch_anc_loss, batch_count = 0, 0, 0, 0
        ensemble.train()

        for batch in train_dataloader:

            optimizer.zero_grad()
            batch["padded_features"] = batch["padded_features"].to(dtype)
            features = batch["padded_features"].to(device)
            encodings, decodings = ensemble(features)
            encodings = [e.to(dtype) for e in encodings]
            decodings = [d.to(dtype) for d in decodings]

            encodings_loss = encodings_criterion(encodings, batch).to(dtype) if encodings_criterion else torch.tensor(0, dtype=dtype)
            decodings_loss = decodings_criterion(decodings, batch).to(dtype) if decodings_criterion else torch.tensor(0, dtype=dtype)
            anchor_loss = anchor_criterion(encodings).to(dtype) if anchor_criterion else torch.tensor(0, dtype=dtype)
	          
            (encodings_loss + decodings_loss + anchor_loss).backward()
            optimizer.step()

            epoch_enc_loss += encodings_loss.item()
            epoch_dec_loss += decodings_loss.item()
            epoch_anc_loss += anchor_loss.item()
            batch_count += 1

        test_epoch_enc_loss, test_epoch_dec_loss, test_epoch_anc_loss, test_batch_count = 0, 0, 0, 0
        ensemble.eval()

        for batch in test_dataloader:
            
            batch["padded_features"] = batch["padded_features"].to(dtype)
            features = batch["padded_features"].to(device)
            encodings, decodings = ensemble(features)
            encodings = [e.to(dtype) for e in encodings]
            decodings = [d.to(dtype) for d in decodings]

            encodings_loss = encodings_criterion(encodings, batch) if encodings_criterion else torch.tensor(0, dtype=dtype)
            decodings_loss = decodings_criterion(decodings, batch) if decodings_criterion else torch.tensor(0, dtype=dtype)
            anchor_loss = anchor_criterion(encodings) if anchor_criterion else torch.tensor(0, dtype=dtype)
	        
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


def train_variational_encoder_ensemble(
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
    show_print=True,
    dtype=DEFAULT_DTYPE
):

    for epoch in range(num_epochs):

        epoch_enc_loss, epoch_dec_loss, epoch_anc_loss, batch_count = 0, 0, 0, 0
        ensemble.train()

        for batch in train_dataloader:

            optimizer.zero_grad()
            features = batch["padded_features"].to(device)
            encodings, zs, decodings = ensemble(features)
            encodings = [e.to(dtype) for e in encodings]
            decodings = [d.to(dtype) for d in decodings]

            encodings_loss = encodings_criterion(encodings, zs, batch) if encodings_criterion else torch.tensor(0, dtype=dtype)
            decodings_loss = decodings_criterion(decodings, batch) if decodings_criterion else torch.tensor(0, dtype=dtype)
            anchor_loss = anchor_criterion(encodings, zs) if anchor_criterion else torch.tensor(0, dtype=dtype)
	          
            (encodings_loss + decodings_loss + anchor_loss).backward()
            optimizer.step()

            epoch_enc_loss += encodings_loss.item()
            epoch_dec_loss += decodings_loss.item()
            epoch_anc_loss += anchor_loss.item()
            batch_count += 1

        test_epoch_enc_loss, test_epoch_dec_loss, test_epoch_anc_loss, test_batch_count = 0, 0, 0, 0
        ensemble.eval()

        for batch in test_dataloader:

            features = batch["padded_features"].to(device)
            encodings, means, decodings = ensemble(features)

            encodings_loss = encodings_criterion(encodings, means, batch) if encodings_criterion else torch.tensor(0, dtype=dtype)
            decodings_loss = decodings_criterion(decodings, batch) if decodings_criterion else torch.tensor(0, dtype=dtype)
            anchor_loss = anchor_criterion(encodings, means) if anchor_criterion else torch.tensor(0, dtype=dtype)
	        
            test_epoch_enc_loss += encodings_loss.item()
            test_epoch_dec_loss += decodings_loss.item()
            test_epoch_anc_loss += anchor_loss.item()
            test_batch_count += 1

        if test_batch_count == 0:
            test_batch_count += 1
        if batch_count == 0:
            batch_count += 1

        if show_print:
            print(
                f"Epoch {epoch}, train enc loss {epoch_enc_loss/batch_count}, train dec loss {epoch_dec_loss/batch_count}, train anc loss {epoch_anc_loss/batch_count}\ntest enc loss {test_epoch_enc_loss/test_batch_count}, test dec loss {test_epoch_dec_loss/test_batch_count}, test anc loss {test_epoch_anc_loss/test_batch_count}",
                flush=True,
            )

        if epoch + 1 in scheduler_epochs:
            scheduler.step()
    
    return ensemble