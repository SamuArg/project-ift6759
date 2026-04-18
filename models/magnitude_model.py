import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.load_dataset import N_INSTRUMENT_CLASSES


class MagnitudePredictor(nn.Module):
    """
    CNN-BiLSTM for predicting earthquake magnitude from a 2-second (200-sample) waveform window.
    Architecture:
      Deep Conv1d blocks -> BiLSTM -> Global Average Pooling -> Deep Linear Regressor
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        lstm_hidden=128,
        dropout=0.3,
        use_coords=False,
        use_vs30=False,
        use_instrument=False,
    ):
        super().__init__()

        self.use_coords = use_coords
        self.use_vs30 = use_vs30
        self.use_instrument = use_instrument

        self.encoder = nn.Sequential(
            # Block 1: 200 -> 100
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2: 100 -> 50
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3: 50 -> 25
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(),
            nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=base_channels * 4,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        regressor_in = lstm_hidden * 2  # sortie bidirectionnelle après pooling
        if self.use_coords:
            regressor_in += 2  # lat + lon
        if self.use_vs30:
            regressor_in += 1  # log10(VS30)
        if self.use_instrument:
            regressor_in += N_INSTRUMENT_CLASSES  # one-hot 6 classes

        # ── Regresseur ───
        self.regressor = nn.Sequential(
            nn.Linear(regressor_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),  # magnitude scalaire
        )

    def forward(
        self,
        x,
        coords=None,
        vs30=None,
        instrument=None,
    ):
        """
        Args:
            x: (B, 3, 200) normalized waveform centered on P-wave
            coords: (B, 2) tensor of latitude and longitude (optional)
        Returns:
            mag: (B, 1) predicted scalar magnitude
        """
        x = self.encoder(x)  # (B, 256, 25)

        x = x.permute(0, 2, 1)  # (B, 25, 256) for LSTM
        x, _ = self.lstm(x)  # (B, 25, 256)

        x = x.mean(dim=1)  # Global Average Pooling over time (B, 256)

        features = [x]

        if self.use_coords:
            if coords is None:
                raise ValueError(
                    "MagnitudePredictor(use_coords=True) mais aucun tensor 'coords' "
                    "n'a été fourni à forward()."
                )
            features.append(coords)  # (B, 2)

        if self.use_vs30:
            if vs30 is None:
                raise ValueError(
                    "MagnitudePredictor(use_vs30=True) mais aucun tensor 'vs30' "
                    "n'a été fourni à forward()."
                )
            features.append(vs30)  # (B, 1)

        if self.use_instrument:
            if instrument is None:
                raise ValueError(
                    "MagnitudePredictor(use_instrument=True) mais aucun tensor "
                    "'instrument' n'a été fourni à forward()."
                )
            features.append(instrument)  # (B, 6)

        if len(features) > 1:
            x = torch.cat(features, dim=1)  # (B, regressor_in)

        mag = self.regressor(x)  # (B, 1)
        return mag


if __name__ == "__main__":
    # Tests de forme — toutes les combinaisons de features
    x = torch.randn(16, 3, 200)
    coords = torch.randn(16, 2)
    vs30 = torch.randn(16, 1)
    instrument = torch.zeros(16, N_INSTRUMENT_CLASSES)
    instrument[:, 0] = 1.0  # HH pour tout le monde

    print("=== Tests de forme MagnitudePredictor ===")
    for uc, uv, ui in [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ]:
        model = MagnitudePredictor(use_coords=uc, use_vs30=uv, use_instrument=ui)
        kwargs = {}
        if uc:
            kwargs["coords"] = coords
        if uv:
            kwargs["vs30"] = vs30
        if ui:
            kwargs["instrument"] = instrument
        out = model(x, **kwargs)
        label = f"coords={uc} vs30={uv} instrument={ui}"
        print(f"  {label:<45} → {out.shape}")  # attendu (16, 1)
