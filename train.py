def train(epoch, use_cuda=False):
    # Load dataset
    root = "/media/data/pymunk_dataset/train/"
    raw_folder_name = "raw"
    processed_folder_name = "processed"
    dataset = BallSimulationDataset(root, raw_folder_name, processed_folder_name, 
                                    use_cuda=use_cuda)
    # Create model
    model = PhysicsVAE(use_cuda=use_cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    # Training
    for e_idx in range(epoch):
        if e_idx >= 2:
            dataset.shuffle()
        train_loss = 0
        for batch_idx, (batch_x, batch_y, edge_index) in enumerate(dataset):
            optimizer.zero_grad()
            recon_batch, z_stats = model(batch_x, batch_y, edge_index)
            loss, reconstr_loss = loss_function(recon_batch, batch_y, z_stats)
            loss.backward()
            train_loss += reconstr_loss.item()
            optimizer.step()
        print(e_idx, train_loss/(batch_idx+1))
    return model

if __name__ == "__main__":
    train(100, use_cuda=True)