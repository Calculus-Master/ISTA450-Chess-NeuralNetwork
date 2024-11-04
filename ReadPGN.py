import numpy as np
import torch
from chess import Color, WHITE, BLACK, PieceType, PAWN, KNIGHT, BISHOP, ROOK, \
    QUEEN, Board
from chess.pgn import read_game, GameNode
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from PytorchNN import Network

db = open("data/AJ-OTB-PGN-001.pgn", encoding="utf-8")

# 10 inputs [White Pawns, Knights, Bishops, Rooks, Queens, Black Pawns, Knights, Bishops, Rooks, Queens]
source_inputs = []
# Output Format: [Chance of White Win, Chance of Black Win, Chance of Stalemate]
source_outputs = []
for i in range(15):
    root: GameNode = read_game(db)
    game_result = root.game().headers["Result"]

    if game_result == "1-0":
        res = [1, 0, 0]
    elif game_result == "0-1":
        res = [0, 1, 0]
    else:
        res = [0, 0, 1]

    node = root
    while not node.is_end():
        source_inputs.append([])
        for col in [WHITE, BLACK]:
            for piece in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN]:
                source_inputs[-1].append(len(node.board().pieces(piece, col)))
        source_outputs.append(res)

        node = node.next()

def label_of_output_vector(output_vector):
    return 0 if output_vector[0] == 1 else 1 if output_vector[1] == 1 else 2

# Dataloader
device = torch.device("mps")

inputs_tensor = torch.tensor(source_inputs, dtype=torch.float32).to(device)
outputs_tensor = torch.tensor([label_of_output_vector(o) for o in source_outputs], dtype=torch.float32).to(device)
dataset = TensorDataset(inputs_tensor, outputs_tensor)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Network training

model = Network().to(device)
model.train_model(train_loader, epochs=12)

model.eval()
with torch.no_grad():
    test_pred = model(inputs_tensor)

model_out = test_pred.cpu().numpy()

print(model_out)