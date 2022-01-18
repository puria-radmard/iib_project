import torch
from config import DEFAULT_DTYPE
device = (
    torch.device('cuda') if torch.cuda.is_available() 
    else torch.device('cpu')
)
from tqdm import tqdm

def audio_regression_script(
        ensemble,
        optimizer,
        scheduler,
        scheduler_epochs,
        criterion,
        train_dataloader,
        test_dataloader,
        num_epochs,
        target_attribute_name,
        is_regression,
        show_print=True,
        dtype=DEFAULT_DTYPE
    ):

    results = []

    for epoch in range(num_epochs):

        epoch_train_loss, batch_count = 0, 0
        ensemble.train()

        for batch in train_dataloader:

            optimizer.zero_grad()
            batch["padded_features"] = batch["padded_features"].to(dtype)
            features = batch["padded_features"].to(device)
            targets = batch[target_attribute_name].to(dtype).to(device)
            _, decodings = ensemble(features)
            decodings = [d.to(dtype) for d in decodings]

            if is_regression:
                decodings = [d.reshape(targets.shape) for d in decodings]
            
            try:
                targets = batch[target_attribute_name].to(dtype).to(device)
                regression_loss = torch.sum(torch.stack([criterion(d, targets) for d in decodings]))
            except:
                targets = batch[target_attribute_name].to(device)
                regression_loss = torch.sum(torch.stack([criterion(d, targets) for d in decodings]))
            
            regression_loss.backward()
            optimizer.step()

            epoch_train_loss += regression_loss.item()
            batch_count += 1

            torch.cuda.empty_cache()

        epoch_test_loss, test_batch_count = 0, 0
        ensemble.eval()

        for batch in test_dataloader:

            batch["padded_features"] = batch["padded_features"].to(dtype)
            features = batch["padded_features"].to(device)
            targets = batch[target_attribute_name].to(device)
            _, decodings = ensemble(features)
            decodings = [d.to(dtype) for d in decodings]

            if is_regression:
                decodings = [d.reshape(targets.shape) for d in decodings]

            try:
                targets = batch[target_attribute_name].to(dtype).to(device)
                test_regression_loss = torch.sum(torch.stack([criterion(d, targets) for d in decodings]))
            except:
                targets = batch[target_attribute_name].to(device)
                test_regression_loss = torch.sum(torch.stack([criterion(d, targets) for d in decodings]))

            epoch_test_loss += test_regression_loss.item()
            test_batch_count += 1

        if test_batch_count == 0:
            test_batch_count += 1
        if batch_count == 0:
            batch_count += 1

        if show_print:
            print(
                f"Epoch {epoch}, train_loss {float(epoch_train_loss/batch_count)}, test_loss {float(epoch_test_loss/test_batch_count)}",
                flush=True,
            )
            
        if epoch + 1 in scheduler_epochs:
            scheduler.step()

        results.append(
            {
                'epoch': epoch,
                'train loss': float(epoch_train_loss/batch_count), 
                'test loss': float(epoch_test_loss/test_batch_count),
            }
        )
    
    return ensemble, results
