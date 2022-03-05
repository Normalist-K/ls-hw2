import os.path as osp
from argparse import ArgumentParser
import psutil
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

from modules import FeatureDataModule, MlpClassifier, FeatureDataset



def parse_args(argv=None):
    parser = ArgumentParser(__file__, add_help=False)
    parser.add_argument('--best_model_path', type=str)
    parser.add_argument('--list_file_path', type=str)
    parser.add_argument('--feature_dir', type=str)
    parser.add_argument('--save_csv_path', type=str)
    args = parser.parse_args(argv)
    return args

def inference(model, device, test_loader):
    model.eval()
    pred_y_list = []
    for data, _ in tqdm(test_loader, total=len(test_loader), desc='Infer', position=0, leave=True):
        data = data.to(device)
        
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=-1)

        pred_y_list.extend(pred.tolist())

    return pred_y_list

def main(args):
    p2_df = pd.read_csv(args.list_file_path)
    dataset = FeatureDataset(df=p2_df, feature_dir=args.feature_dir, p2=True)
    dataloader = DataLoader(dataset, batch_size=512, drop_last=False, shuffle=False,
                            pin_memory=True, num_workers=len(psutil.Process().cpu_affinity()))
    model = MlpClassifier.load_from_checkpoint(args.best_model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    preds = inference(model.to(device), device, dataloader)
    assert len(preds) == len(p2_df)
    df = p2_df.copy()
    df['Category'] = preds
    df.to_csv(args.save_csv_path, index=False)
    print('Output file:', args.save_csv_path)

if __name__ == '__main__':
    main(parse_args())
