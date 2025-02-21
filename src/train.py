import torch


def train_model(model, dataloaders, criterion, optimizer, num_epochs=50, device='cpu'):
    """
    Train the model and return the best model.

    Parameters:
        model: PyTorch model
        dataloaders: dict with keys 'train' and 'val' and values DataLoader objects
        criterion: loss function
        optimizer: optimization algorithm
        num_epochs: number of epochs
        device: 'cpu' or 'cuda'
    
    Returns:
        best_model: PyTorch model with best validation accuracy
    """
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()   # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for audio_inputs, landmark_inputs, labels in dataloaders[phase]:
                audio_inputs = audio_inputs.to(device)
                landmark_inputs = landmark_inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(audio_inputs, landmark_inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * audio_inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(dataloaders[phase].sampler)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].sampler)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model if validation accuracy improves
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f"Best Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model