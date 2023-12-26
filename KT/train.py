import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def inference(features, questions, skills, labels, seq_len, model, loss_func):
    if model._get_name() == 'DKT':
        pred = model(features, questions, skills, labels)
        loss_kt, auc, acc = loss_func(pred, labels)

    elif model._get_name() == 'DKT_AUG':
        pred = model(features, questions, labels, seq_len)
        assert torch.all(pred > 0)
        loss_kt, auc, acc = loss_func(pred, labels)
    elif model._get_name() == 'DKT_PEBG':
        pred = model(questions, labels)
        loss_kt, auc, acc = loss_func(pred, labels)
        # l2_lambda = 0.00001
        # l2_norm = sum(p.pow(2.0).sum()
        #               for p in model.parameters())
        # loss_kt = loss_kt + l2_lambda * l2_norm
    elif model._get_name() == 'DKT_PLUS':
        pred = model(questions, labels)
        loss_kt, auc, acc = loss_func(pred, labels)
        # l2_lambda = 0.00001
        # l2_norm = sum(p.pow(2.0).sum()
        #               for p in model.parameters())
        # loss_kt = loss_kt + l2_lambda * l2_norm
    elif model._get_name() == 'QGKT':

        pred = model(features, questions, labels)

        loss_kt, auc, acc = loss_func(pred, labels)
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum()
                      for p in model.parameters())

        loss_kt = loss_kt + l2_lambda * l2_norm

    elif model._get_name() in {'DKVMN', 'DKVMN_RE'}:
        loss_kt, filtered_pred, filtered_target = model.forward(questions, features, labels.reshape(-1, 1))
        from sklearn.metrics import roc_auc_score, accuracy_score
        filtered_target = filtered_target.cpu().detach().numpy()
        filtered_pred = filtered_pred.cpu().detach().numpy()
        auc = roc_auc_score(filtered_target, filtered_pred)
        filtered_pred[filtered_pred >= 0.5] = 1.0
        filtered_pred[filtered_pred < 0.5] = 0.0
        acc = accuracy_score(filtered_target, filtered_pred)
        exit(-1)
    elif model._get_name() == 'GKT':
        pred, ec_list, rec_list, z_prob_list = model(features, questions)
        loss_kt, auc, acc = loss_func(pred, labels)
    elif model._get_name() in {'SAKT', 'SAKT_SKILL'}:
        input_len = features.shape[1]
        batch_size = features.shape[0]

        if input_len < model.seq_len:
            pad_len = model.seq_len - input_len
            features = torch.cat((features, -1 + torch.zeros([batch_size, pad_len])), dim=1)
            questions = torch.cat((questions, -1 + torch.zeros([batch_size, pad_len])), dim=1)

            labels = torch.cat((labels, -1 + torch.zeros([batch_size, pad_len])), dim=1)
        features = features[:, :model.seq_len].long()
        questions = questions[:, :model.seq_len].long()
        labels = labels[:, :model.seq_len]

        pred = model(features, questions, labels)

        loss_kt, auc, acc = loss_func(pred, labels)


    elif model._get_name() == 'AKT':
        s_num = model.s_num
        correct = (labels == 1)
        features = torch.IntTensor(skills.cpu().numpy() + correct.cpu().numpy() * s_num).to(skills.device)

        pred, c_loss = model(
            skills,
            features,
            questions
        )
        loss_kt, auc, acc = loss_func(pred, labels)
    else:
        loss_kt = None
        auc = None
        acc = None

    return loss_kt, auc, acc


def train_epoch(model: nn.Module, data_loader, optimizer, loss_func, logs, cuda):
    model.train()
    loss_train = []
    auc_train = []
    acc_train = []
    for batch in tqdm(data_loader, desc="Training batches", leave=False, ncols=80):
        optimizer.zero_grad()
        features, questions, skills, labels, seq_len = batch

        if cuda:
            features, questions, skills, labels = features.cuda(), questions.cuda(), skills.cuda(), labels.cuda()

        loss_kt, auc, acc = inference(features, questions, skills, labels, seq_len, model, loss_func)
        loss_train.append(loss_kt.cpu().detach().numpy())
        auc_train.append(auc)
        acc_train.append(acc)

        # print('batch idx: ', batch_idx, 'loss kt: ', loss_kt.item(), 'auc: ', auc, 'acc: ', acc)

        loss_kt.backward()
        optimizer.step()
        # torch.nn.utils.clip_grad_norm_(
        #     model.parameters(), max_norm=params.maxgradnorm)

    print('loss_train: {:.10f}'.format(np.mean(loss_train)),
          'auc_train: {:.10f}'.format(np.mean(auc_train)),
          'acc_train: {:.10f}'.format(np.mean(acc_train))
          )
    logs['train_loss'].append(np.mean(loss_train))
    logs['train_auc'].append(np.mean(auc_train))
    logs['train_acc'].append(np.mean(acc_train))

    return model, optimizer


def val_epoch(model: nn.Module, data_loader, optimizer, loss_func, logs, cuda=False):
    model.eval()
    loss_val = []
    auc_val = []
    acc_val = []

    save_dir = '/Users/watsonyang/PycharmProjects/MyKT/save'
    model_file = save_dir + '/' + 'model.pt'
    optimizer_file = save_dir + '/' + 'opt.pt'

    with torch.no_grad():

        for batch_idx, batch in tqdm(enumerate(data_loader), desc="Validating batches", unit="iteration", ncols=80):
            features, questions, skills, labels, seq_len = batch
            if cuda:
                features, questions, skills, labels = features.cuda(), questions.cuda(), skills.cuda(), labels.cuda()

            loss_kt, auc, acc = inference(features, questions, skills, labels, seq_len, model, loss_func)
            loss_val.append(loss_kt.cpu())
            auc_val.append(auc)
            acc_val.append(acc)

        print('loss_val: {:.10f}'.format(np.mean(loss_val)),
              'auc_val: {:.10f}'.format(np.mean(auc_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val))
              )
        logs['val_loss'].append(np.mean(loss_val))
        logs['val_auc'].append(np.mean(auc_val))
        logs['val_acc'].append(np.mean(acc_val))


def train(model: nn.Module, data_loaders, optimizer, loss_func, args):
    # train loop
    metrics = ['train_auc', 'train_loss', 'train_acc', 'val_auc', 'val_loss', 'val_acc']
    if hasattr(args, 'logs'):
        logs = args.logs
    else:
        logs = {}
        for metric in metrics:
            logs[metric] = []
    best_val_auc = -1
    best_epoch = -1
    early_stop = True
    early_stop_interval = 25
    if args.current_epoch != 0:
        print(f"start training from previous checkpoints of epoch {args.current_epoch}")
    for epoch in range(args.current_epoch + 1, args.current_epoch + args.n_epochs):
        print(f"epoch: {epoch}")

        train_epoch(model, data_loader=data_loaders['train_loader'], optimizer=optimizer,
                    loss_func=loss_func, logs=logs, cuda=args.cuda)

        val_epoch(model, data_loader=data_loaders['test_loader'], optimizer=optimizer,
                  loss_func=loss_func, logs=logs, cuda=args.cuda)

        if max(logs['val_auc']) > best_val_auc:
            best_val_auc = max(logs['val_auc'])
            best_epoch = epoch
            print(f'Epoch: {epoch} Best Test Set AUC Updated {best_val_auc}')
            """
                We are going to save checkpoints here
                
                1. checkpoint path
                2. checkpoint file name
            """

            from KT.util.checkpoint import CheckpointManager

            CheckpointManager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                dataset=args.dataset,
                epoch=epoch,
                model_name=args.model,
                hyperparameters=model.get_hyperparameters(),
            )
            # ToDo : update to save metrics record as well

        if early_stop:
            if best_epoch + early_stop_interval < epoch:
                print(f'Early Stop at epoch {epoch}!')
                args.n_epochs = epoch
                print(f'Epoch: {epoch} Best Test Set AUC Updated {best_val_auc}')
                break
    for metric in metrics:
        arr = np.array(logs[metric])
        if metric in {'train_loss', 'val_loss'}:
            best_record_index = np.argmin(arr)
        else:
            best_record_index = np.argmax(arr)

        output = f"Best {metric} in epoch {best_record_index}: "
        for m in metrics:
            output += m + ":  " + f"{logs[m][best_record_index]}  "
        print(output)

    import matplotlib.pyplot as plt

    for m in metrics:
        plt.plot(np.arange(args.current_epoch, args.current_epoch + args.n_epochs, dtype=int), logs[m], label=m)
    plt.legend(loc="upper left", shadow=True, title=model._get_name(), fancybox=True)
    plt.show()
    # plot some figures here
    return logs


def test(model: nn.Module, data_loader, optimizer, loss_func, n_epochs, cuda=True):
    model.eval()
    hidden_list = []
    # ToDo: wtf does this function do?
    with torch.no_grad():

        for batch_idx, (features, questions, answers) in enumerate(data_loader):
            if cuda:
                features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
            hidden = model._get_hidden(features, questions)
            hidden = hidden[:, -1, :]
            hidden = torch.squeeze(hidden)

            hidden_list.append(hidden)

    hidden_total = torch.cat(hidden_list, 0).numpy()
    # kmeans = KMeans(n_clusters=4,random_state=0).fit(hidden_total)

    from sklearn.manifold import TSNE

    X_embedding = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(hidden_total)

    import matplotlib.pyplot as plt

    plt.scatter(X_embedding[:, 0], X_embedding[:, 1])
    plt.show()

    return hidden_total


def evaluate_akt(model: nn.Module, data_loaders, loss_func, cuda=False):
    # evaluate the train dataset
    loss_train = []
    auc_train = []
    acc_train = []
    s_num = model.s_num

    '''
        generating sequences to observe relations between skills
    '''
    q_custom = []
    y_custom = []
    f_custom = []
    for si in range(1, s_num + 1):
        for sj in range(1, s_num + 1):
            for ai in range(0, 2):
                for aj in range(0, 2):
                    fi = s_num * ai + si
                    fj = s_num * aj + sj
                    q_custom.append([si, sj])
                    y_custom.append([ai, aj])
                    f_custom.append([fi, fj])

    q_custom = torch.IntTensor(q_custom)
    y_custom = torch.FloatTensor(y_custom)
    f_custom = torch.IntTensor(f_custom)

    print(q_custom)
    print(q_custom.shape)
    custom_size = q_custom.shape[0]

    max_seq_len = 200
    pad_len = max_seq_len - q_custom.shape[1]

    device = 'cuda:0'

    q_custom = torch.cat([q_custom, torch.zeros([custom_size, pad_len], dtype=int)], dim=1).to(device)
    y_custom = torch.cat([y_custom, -1 * torch.ones([custom_size, pad_len], dtype=float)], dim=1).to(device)
    f_custom = torch.cat([f_custom, torch.zeros([custom_size, pad_len], dtype=int)], dim=1).to(device)

    import matplotlib.pyplot as plt
    co_relavance_upgrade = torch.zeros([s_num + 1, s_num + 1])
    co_relavance_downgrade = torch.zeros([s_num + 1, s_num + 1])

    def demo(matrix1, matrix2, matrix3):
        # Create subplots with 1 row and 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the first matrix
        im1 = axes[0].imshow(matrix1, cmap='viridis')
        axes[0].set_title('Matrix 1')

        # Plot the second matrix
        im2 = axes[1].imshow(matrix2, cmap='plasma')
        axes[1].set_title('Matrix 2')

        # Plot the third matrix
        im3 = axes[2].imshow(matrix3, cmap='inferno')
        axes[2].set_title('Matrix 3')

        # Add colorbars
        fig.colorbar(im1, ax=axes[0])
        fig.colorbar(im2, ax=axes[1])
        fig.colorbar(im3, ax=axes[2])

        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the plot
        plt.show()

    custom_bs = 10
    batch_idx = 0
    pred = None
    while batch_idx < q_custom.shape[0]:
        low = batch_idx
        high = min(batch_idx + custom_bs, q_custom.shape[0])
        q_in = q_custom[low:high, ]
        f_in = f_custom[low:high, ]
        y_in = y_custom[low:high, ]

        pred_batch, _ = model(q_in, f_in, q_in)
        batch_idx += custom_bs
        pred = pred_batch if pred is None else torch.cat([pred, pred_batch], dim=0)
    print(pred.shape)
    exit(-1)
    for i in range(s_num ** 2):
        qi = q_custom[4 * i:4 * i + 4, :]
        fi = f_custom[4 * i:4 * i + 4, :]
        yi = y_custom[4 * i:4 * i + 4, :]
        pred, _ = model(qi, fi, qi)
        si = qi[0, 0].cpu().item()
        sj = qi[0, 1].cpu().item()
        print('question seq: ', qi[:, :4])
        print('feature seq', fi[:, :4])
        print('label seq', yi[:, :4])
        co_relavance_downgrade[si, sj] = pred[0, 0]
        co_relavance_upgrade[si, sj] = pred[2, 0]
        print(
            f'co_relavance for {[si, sj]}: update {co_relavance_upgrade[si, sj]} downgrade{co_relavance_downgrade[si, sj]}')

    demo(co_relavance_downgrade.numpy(), co_relavance_upgrade.numpy(),
         co_relavance_upgrade.numpy() - co_relavance_downgrade.numpy())

    print(f'nums of skill {s_num}')
    for dataset_type, data_loader in data_loaders.items():
        print(f'evaluating AKT on dataset : {dataset_type}')

        # for batch_idx, batch in enumerate(data_loader):
        #     features, questions, skills, labels, seq_len = batch
        #     batch_size = features.shape[0]
        #     if cuda:
        #         features, questions, skills, labels = features.cuda(), questions.cuda(), skills.cuda(), labels.cuda()
        #
        #     pred,_ = model(questions, features, questions)
        #     print(pred)
        #     '''
        #         question[i,0:j]
        #         labels[i,0:j]
        #         pred[i,j] : correct probability for questions[i,j+1]
        #
        #         evaluate the sequence into
        #     '''
        #
        #     for i in range(batch_size):
        #         seq_len_i = seq_len[i]
        #         print('-------evaluating new sequence------------')
        #         for j in range(0,seq_len_i-1):
        #             print('-------------------')
        #             print(f'q seq {questions[i,0:j].cpu().tolist()}')
        #             print(f'a seq {labels[i,0:j].cpu().tolist()}')
        #             print(f'f seq {features[i,0:j].cpu().tolist()}')
        #             print(f'next q {questions[i,j+1]}')
        #             print(f'prediction:{pred[i,j]}, true label:{labels[i,j+1]}')
        #
        #
        #     loss_kt, auc, acc = loss_func(pred, labels)
        #     print(loss_kt)
        #     print(auc)
        #     print(acc)

    # evaluate the test dataset


def evaluate_sakt(model: nn.Module, data_loaders, loss_func, cuda=False):
    s_num = 117
    q_custom = []
    y_custom = []
    f_custom = []
    n = 1
    for si in range(1, s_num + 1):
        for sj in range(1, s_num + 1):
            for ai in range(0, 2):
                for aj in range(0, 2):
                    fi = s_num * ai + si
                    fj = s_num * aj + sj
                    q_custom.append([si] * n + [sj] * n + [si] * n)
                    y_custom.append([0] * n + [1] * n + [1] * n)
                    f_custom.append([fi] * n + [fj] * n + [fi] * n)

    q_custom = torch.IntTensor(q_custom)
    y_custom = torch.FloatTensor(y_custom)
    f_custom = torch.IntTensor(f_custom)

    print(q_custom)
    print(q_custom.shape)
    custom_size = q_custom.shape[0]

    max_seq_len = 200
    pad_len = max_seq_len - q_custom.shape[1]

    device = 'cuda:0'

    q_custom = torch.cat([q_custom, torch.zeros([custom_size, pad_len], dtype=int)], dim=1).to(device)
    y_custom = torch.cat([y_custom, -1 * torch.ones([custom_size, pad_len], dtype=float)], dim=1).to(device)
    f_custom = torch.cat([f_custom, torch.zeros([custom_size, pad_len], dtype=int)], dim=1).to(device)

    import matplotlib.pyplot as plt
    co_relavance_upgrade = torch.zeros([s_num + 1, s_num + 1])
    co_relavance_downgrade = torch.zeros([s_num + 1, s_num + 1])

    def demo(matrix1, matrix2, matrix3):
        # Create subplots with 1 row and 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the first matrix
        im1 = axes[0].imshow(matrix1, cmap='viridis', origin='lower')
        axes[0].set_title('Matrix 1')

        # Plot the second matrix
        im2 = axes[1].imshow(matrix2, cmap='plasma', origin='lower')
        axes[1].set_title('Matrix 2')

        # Plot the third matrix
        im3 = axes[2].imshow(matrix3, cmap='inferno', origin='lower')
        axes[2].set_title('Matrix 3')

        # Add colorbars
        fig.colorbar(im1, ax=axes[0])
        fig.colorbar(im2, ax=axes[1])
        fig.colorbar(im3, ax=axes[2])

        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the plot
        plt.show()

    custom_bs = 10
    batch_idx = 0
    pred = None
    while batch_idx < q_custom.shape[0]:
        low = batch_idx
        high = min(batch_idx + custom_bs, q_custom.shape[0])
        q_in = q_custom[low:high, ]
        f_in = f_custom[low:high, ]
        y_in = y_custom[low:high, ]

        pred_batch = model(f_in, q_in, y_in)
        batch_idx += custom_bs
        pred = pred_batch if pred is None else torch.cat([pred, pred_batch], dim=0)
    print(pred.shape)

    for i in range(s_num ** 2):
        qi = q_custom[4 * i:4 * i + 4, :]
        fi = f_custom[4 * i:4 * i + 4, :]
        yi = y_custom[4 * i:4 * i + 4, :]

        si = qi[0, n - 1].cpu().item()
        sj = qi[0, 2 * n - 1].cpu().item()
        # print('question seq: ', qi[:, :3 * n])
        # print('feature seq', fi[:, :3 * n])
        # print('label seq', yi[:, :3 * n])
        #
        # print('next q seq', qi[:, 1:3 * n])
        # print('pred seq', pred[4 * i:4 * i + 4, :3 * n - 1])
        co_relavance_upgrade[si, sj] = pred[4 * i + 1, 2 * n - 1] - pred[4 * i + 1, n - 1]
        co_relavance_downgrade[si, sj] = -(pred[4 * i + 2, 2 * n - 1] - pred[4 * i + 2, n - 1])

        print(
            f'co_relavance for {[si, sj]}: update {co_relavance_upgrade[si, sj]} downgrade{co_relavance_downgrade[si, sj]}')

    sorted_indices = np.argsort(co_relavance_upgrade.flatten())
    dim1, dim2 = np.unravel_index(sorted_indices, co_relavance_upgrade.shape)

    for _i in range(len(dim1)):
        i, j = dim1[_i], dim2[_i]
        print(f'({i},{j}) : {co_relavance_upgrade[i, j]}')
    exit(-1)
    demo(co_relavance_downgrade.numpy(), co_relavance_upgrade.numpy(),
         co_relavance_upgrade.numpy() - co_relavance_downgrade.numpy())

    for dataset_type, data_loader in data_loaders.items():
        print(f'evaluating dataset : {dataset_type}')

        loss_train = []
        auc_train = []
        acc_train = []

        for batch_idx, batch in enumerate(data_loader):
            features, questions, skills, labels, seq_len = batch
            if cuda:
                features, questions, skills, labels = features.cuda(), questions.cuda(), skills.cuda(), labels.cuda()
            pred = model(questions, features, labels)

            loss_kt, auc, acc = loss_func(pred, labels)
            print(loss_kt)
            print(auc)
            print(acc)
    # input_len = features.shape[1]
    # batch_size = features.shape[0]
    #
    # if input_len < model.seq_len:
    #     pad_len = model.seq_len - input_len
    #     features = torch.cat((features, -1 + torch.zeros([batch_size, pad_len])), dim=1)
    #     questions = torch.cat((questions, -1 + torch.zeros([batch_size, pad_len])), dim=1)
    #
    #     labels = torch.cat((labels, -1 + torch.zeros([batch_size, pad_len])), dim=1)
    # features = features[:, :model.seq_len].long()
    # questions = questions[:, :model.seq_len].long()
    # labels = labels[:, :model.seq_len]
    # if model._get_name() == 'SAKT':
    #     pred = model(features, questions)
    # else:
    #     pred = model(features, questions, labels)
    #
    # loss_kt, auc, acc = loss_func(pred, labels)


def evaluate_sakt_skill(model: nn.Module, data_loaders, loss_func, cuda=False):
    for dataset_type, data_loader in data_loaders.items():
        print(f'evaluating SAKT-Skill on dataset : {dataset_type}')

        loss_train = []
        auc_train = []
        acc_train = []

        assert data_loader.batch_size == 1
        for batch_idx, batch in enumerate(data_loader):

            features, questions, skills, labels, seq_len = batch
            if cuda:
                features, questions, skills, labels = features.cuda(), questions.cuda(), skills.cuda(), labels.cuda()

            pred = model(features, questions, labels)
            assert torch.all(pred > 0)
            pred_copy = pred.clone()
            loss_kt, auc, acc = loss_func(pred, labels)

            def print_result(q, s, y, l, p, loss, auc, acc):
                print(l)

                q = q.squeeze().cpu().tolist()
                s = s.squeeze().cpu().tolist()
                y = y.squeeze().cpu().tolist()
                l = l.squeeze().cpu().tolist()
                p = p.squeeze().cpu().tolist()
                print(f'question seq: {q[:l]}')
                print(f'skill seq: {s[:l]}')
                print(f'label seq {y[:l]}')

                print(f'pred seq:{p[:l]}')

                print(f'loss {loss} auc {auc} acc {acc}')

            print_result(questions, skills, labels, seq_len, pred_copy, loss_kt, auc, acc)


def evaluate(model: nn.Module, data_loaders, loss_func, cuda=False):
    model.eval()

    with torch.no_grad():
        if model._get_name() == 'AKT':
            evaluate_akt(model, data_loaders, loss_func, cuda)

        if model._get_name() == 'SAKT_SKILL':
            evaluate_sakt_skill(model, data_loaders, loss_func, cuda)

        if model._get_name() == 'SAKT':
            evaluate_sakt(model, data_loaders, loss_func, cuda)


from torch import cuda


def get_less_used_gpu(gpus=None, debug=True):
    """Inspect cached/reserved and allocated memory on specified gpus and return the id of the less used device"""
    if gpus is None:
        warn = 'Falling back to default: all gpus'
        gpus = range(cuda.device_count())
    elif isinstance(gpus, str):
        gpus = [int(el) for el in gpus.split(',')]

    # check gpus arg VS available gpus
    sys_gpus = list(range(cuda.device_count()))
    if len(gpus) > len(sys_gpus):
        gpus = sys_gpus
        warn = f'WARNING: Specified {len(gpus)} gpus, but only {cuda.device_count()} available. Falling back to default: all gpus.\nIDs:\t{list(gpus)}'
    elif set(gpus).difference(sys_gpus):
        # take correctly specified and add as much bad specifications as unused system gpus
        available_gpus = set(gpus).intersection(sys_gpus)
        unavailable_gpus = set(gpus).difference(sys_gpus)
        unused_gpus = set(sys_gpus).difference(gpus)
        gpus = list(available_gpus) + list(unused_gpus)[:len(unavailable_gpus)]
        warn = f'GPU ids {unavailable_gpus} not available. Falling back to {len(gpus)} device(s).\nIDs:\t{list(gpus)}'

    cur_allocated_mem = {}
    cur_cached_mem = {}
    max_allocated_mem = {}
    max_cached_mem = {}
    for i in gpus:
        cur_allocated_mem[i] = cuda.memory_allocated(i)
        cur_cached_mem[i] = cuda.memory_reserved(i)
        max_allocated_mem[i] = cuda.max_memory_allocated(i)
        max_cached_mem[i] = cuda.max_memory_reserved(i)
    min_allocated = min(cur_allocated_mem, key=cur_allocated_mem.get)
    if debug:
        print(warn)
        print('Current allocated memory:', {f'cuda:{k}': v for k, v in cur_allocated_mem.items()})
        print('Current reserved memory:', {f'cuda:{k}': v for k, v in cur_cached_mem.items()})
        print('Maximum allocated memory:', {f'cuda:{k}': v for k, v in max_allocated_mem.items()})
        print('Maximum reserved memory:', {f'cuda:{k}': v for k, v in max_cached_mem.items()})
        print('Suggested GPU:', min_allocated)
    return min_allocated


def free_memory(to_delete: list, debug=False):
    import gc
    import inspect
    calling_namespace = inspect.currentframe().f_back
    if debug:
        print('Before:')
        get_less_used_gpu(debug=True)

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        cuda.empty_cache()
    if debug:
        print('After:')
        get_less_used_gpu(debug=True)


def rank_data_performance(model, data_loader, loss_func, cuda=True):
    # in this part we want to rank all data by performance
    # we have to re-write a loss function to do this
    metric_list = {'auc': [], 'loss': [], 'acc': []}
    from torch.utils.data import DataLoader, Subset
    subset_index = [i for i in range(0, 5000)]

    # new_data_loader = DataLoader(data_loader.dataset,batch_size=1,shuffle=False)
    new_data_loader = DataLoader(Subset(data_loader.dataset, subset_index), batch_size=1, shuffle=False)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(new_data_loader):

            features, questions, skills, labels, seq_len = batch

            if cuda:
                features, questions, skills, labels = features.cuda(), questions.cuda(), skills.cuda(), labels.cuda()
            loss, auc, acc = inference(features, questions, skills, labels, seq_len, model, loss_func)
            seq_len = seq_len.squeeze()

            metric_list['loss'].append(
                (loss.cpu(), questions.squeeze().tolist()[:seq_len], labels.squeeze().tolist()[:seq_len]))
            metric_list['auc'].append(
                (auc, questions.squeeze().tolist()[:seq_len], labels.squeeze().tolist()[:seq_len]))
            metric_list['acc'].append(
                (acc, questions.squeeze().tolist()[:seq_len], labels.squeeze().tolist()[:seq_len]))
            del features, questions, skills, labels
            del loss

            torch.cuda.empty_cache()

    metric_list['loss'] = sorted(metric_list['loss'], key=lambda x: x[0])
    metric_list['auc'] = sorted(metric_list['auc'], key=lambda x: x[0], reverse=True)
    metric_list['acc'] = sorted(metric_list['acc'], key=lambda x: x[0], reverse=True)

    return metric_list
