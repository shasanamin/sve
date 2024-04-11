from .subset_trainer import *
from scipy.stats import entropy
from scipy.special import softmax

# using prediction vector entropy (H(y^)) weighted random sampling
class RandETrainer(SubsetTrainer):
    def __init__(
        self, 
        args: argparse.Namespace,
        model: nn.Module,
        train_dataset: IndexedDataset,
        val_loader: DataLoader,
        train_weights: torch.Tensor = None,
    ):
        super().__init__(args, model, train_dataset, val_loader, train_weights)

    def _select_subset(self, epoch, training_steps):
        # select a subset of the data
        self.num_selection += 1

        # too much noise at start (model has extremely low accuracy everywhere)
        warm_start = False
        epochs_cold = 20
        if warm_start and epoch<epochs_cold:
            self.subset = np.random.choice(
                len(self.train_dataset), 
                size=int(len(self.train_dataset) * self.args.train_frac),
                # make sure to pass replacement parameter
                replace=False
            )
        
        else:
            # unshuffled entire training data
            train_val_loader = DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True
            )

            self.model.eval()
            train_output = np.zeros((len(self.train_dataset), self.args.num_classes))
            train_softmax = np.zeros((len(self.train_dataset), self.args.num_classes))
            pred_entropy = np.zeros(len(self.train_dataset))
            with torch.no_grad():
                for _, (data, _, data_idx) in enumerate(train_val_loader):
                    data = data.to(self.args.device)
                    output = self.model(data)
                    train_output[data_idx] = output.cpu().numpy()
                    train_softmax[data_idx] = output.softmax(dim=1).cpu().numpy()
                    pred_entropy[data_idx] = entropy(train_softmax[data_idx], axis=1)
            self.model.train()

            self.subset = np.random.choice(
                len(self.train_dataset), 
                size=int(len(self.train_dataset) * self.args.train_frac),
                replace=self.args.sample_with_rep,
                p=softmax(pred_entropy)
            )

        self.subset_weights = np.ones(len(self.subset))
    