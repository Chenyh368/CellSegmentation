import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.append("./")
from torch.utils.data import DataLoader
from utils.scs import dir_to_class
import torch
from options import Options, HParams
from utils.experiman import ExperiMan
from data import find_dataset_using_name


def main(opt, manager):
    assert torch.cuda.is_available(), "CPU training is not allowed."
    logger = manager.get_logger()
    logger.info(f"======> Single GPU Training")
    # Set up tensorboard
    manager._third_party_tools = ('tensorboard',)
    manager._setup_third_party_tools()

    # Create dataset
    dataset_all_patches = find_dataset_using_name(dataset_name=opt.dataset.name)(opt, manager)
    logger.info("======> dataset [%s] was created" % type(dataset_all_patches).__name__)
    logger.info(f"======> length of dataset: {len(dataset_all_patches)}")

    # Method
    logger.info(f"======> Model: "+ opt.model.name)
    if opt.model.name == "scs":
        from models.scs_model import SCSModel as Model
    else:
        raise NotImplementedError

    logger.info(f"======> creating model")
    model = Model(
        opt = opt,
        manager = manager,
        dataloader = None
    ).cuda()

    # =========== Optimization Setting =============
    # Already set in the model

    # ============= Training ===================
    logger.info(f"======> Begin training")
    total_num_steps = 0
    for epoch_num in range(opt.training.num_epochs):
        for p_index in range(len(dataset_all_patches)):
            per_patch_dataset = dataset_all_patches[p_index]
            per_patch_loader = DataLoader(
                per_patch_dataset,
                batch_size = opt.training.batch_size,
                num_workers = 1,
                shuffle=True,
            )

            for x_train, x_train_pos, y_train, y_binary_train in per_patch_loader:
                for i in range(len(y_train)):
                    if y_train[i][0] != -1:
                        y_train[i] = y_train[i] - x_train_pos[i][0]
                    else:
                        y_train[i][0] = -9999
                        y_train[i][1] = -9999
                y_train = dir_to_class(y_train, opt.model.arch.class_num)
                losses, predictions = model.train_step({
                    'x': [x_train, x_train_pos],
                    'y': [y_train, y_binary_train]
                })
                total_num_steps += 1
                logger.info(f"iter {total_num_steps}| {losses}")


if __name__ == "__main__":
    manager = ExperiMan(name='default')
    parser = manager.get_basic_arg_parser()
    opt = Options(parser).parse()  # get training options

    manager.setup(opt)
    opt = HParams(**vars(opt))
    main(opt, manager)