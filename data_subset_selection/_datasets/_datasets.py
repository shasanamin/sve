import os
import torchvision
import torchvision.transforms as transforms


def get_dataset(args, train=True, train_transform=True):
    if args.dataset in ['cifar10', 'cifar10_1dp', 'cifar10_10dp', 'cifar100', 'cifar100_1dp', 'cifar100_10dp']:
        if (args.dataset == 'cifar10') or ('cifar10_' in args.dataset):
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif (args.dataset == 'cifar100') or ('cifar100_' in args.dataset):
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        else:
            raise NotImplementedError

        if train and train_transform:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        dataset_name = args.dataset.split('_')[0].upper()
        dataset = torchvision.datasets.__dict__[dataset_name](
            root=args.data_dir, train=train, 
            transform=transform, download=True)

        if train and ('dp' in args.dataset):
            # flip labels
            if '_10dp' in args.dataset:
                pct_poison = 10
            else:
                pct_poison = 1
                
            print(f"Flipping {pct_poison} percent training labels")
            import numpy as np
            Y_train = np.array(dataset.targets)
            num_poison = int((pct_poison/100) * len(Y_train))
            ix_poison = np.random.choice(len(Y_train), num_poison, replace=False)

            Y_train_dp = Y_train.copy()
            NUM_CLASSES = np.unique(Y_train).shape[0]
            for i in ix_poison:
                y_curr = Y_train[i]
                y_new = np.random.choice([y for y in range(NUM_CLASSES) if y != y_curr])
                Y_train_dp[i] = y_new

            ix_poisoned = np.zeros(len(Y_train)).astype(int)
            ix_poisoned[ix_poison] = 1

            dataset.targets = list(Y_train_dp)
            # print((np.array(dataset.targets)==Y_train).mean())
            ### eventually add for analysis
            # may impact dataset format and lead to failure in standard pipeline
            # dataset.poisoned = list(ix_poisoned)
            

            
    elif args.dataset == 'tinyimagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        
        dirname = 'tiny-imagenet-200' 
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])      

        if train:
            data_dir = os.path.join(args.data_dir, f'{dirname}/train')
        else:
            data_dir = os.path.join(args.data_dir, f'{dirname}/val')

        dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
    ###
    elif args.dataset == 'snli':
        from datasets import load_dataset
        from transformers import RobertaTokenizer

        dataset = load_dataset("snli")
        
        # Validate labels
        labels = lambda sample: sample['label'] != -1
        dataset = dataset.filter(labels)
        
        dataset = dataset['train'] if train else dataset['test']
        ### TEMP: For faster pipeline setup validation
        # dataset = dataset['validation'] if train else dataset['test']
        ###

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        max_token_length = 0
        for sample in dataset:
            premise_tokens = tokenizer.tokenize(sample['premise'])
            hypothesis_tokens = tokenizer.tokenize(sample['hypothesis'])
            total_tokens = len(premise_tokens) + len(hypothesis_tokens)
            max_token_length = max(max_token_length, total_tokens)

        tokenize_function = lambda data: tokenizer(data['premise'], data['hypothesis'], padding='max_length', truncation=True, max_length=max_token_length)
        dataset = dataset.map(tokenize_function, batched=True)
    ###

    else:
        raise NotImplementedError(f'Unknown dataset: {args.dataset}')

    return dataset