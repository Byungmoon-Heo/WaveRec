import torch

class WaveletFamily:
    def __init__(self, filter_type, pass_weight, filter_length, sigma):
        """
        Initialize the WaveletFamily with a specific filter type and pass weight.
        :param filter_type: Type of wavelet filter ('haar', 'db2', etc.)
        :param pass_weight: Scaling factor for the filters
        """
        self.filter_type = filter_type
        self.pass_weight = pass_weight
        self.filter_length = filter_length
        self.sigma = sigma

    def generate_filters(self):
        """
        Generate lowpass and highpass filters based on the selected wavelet type.
        :return: lowpass_filter, highpass_filter (torch.Tensor)
        """
        if self.filter_type == "haar":
            lowpass = torch.tensor([self.pass_weight, self.pass_weight], dtype=torch.float32)
            highpass = torch.tensor([self.pass_weight, -self.pass_weight], dtype=torch.float32)

        elif self.filter_type == "db2":
            sqrt3 = torch.sqrt(torch.tensor(3.0, dtype=torch.float32))
            norm_factor = torch.sqrt(torch.tensor(2.0, dtype=torch.float32))  # Normalization factor
            lowpass = torch.tensor(
                [
                    (1 + sqrt3) / 4 * self.pass_weight,
                    (3 + sqrt3) / 4 * self.pass_weight,
                    (3 - sqrt3) / 4 * self.pass_weight,
                    (1 - sqrt3) / 4 * self.pass_weight,
                ],
                dtype=torch.float32,
            )
            highpass = torch.tensor(
                [
                    (1 - sqrt3) / 4 * self.pass_weight,
                    -(3 - sqrt3) / 4 * self.pass_weight,
                    (3 + sqrt3) / 4 * self.pass_weight,
                    -(1 + sqrt3) / 4 * self.pass_weight,
                ],
                dtype=torch.float32,
            )
            # Normalize db2
            lowpass = lowpass / norm_factor
            highpass = highpass / norm_factor

        elif self.filter_type == "coiflet":
            lowpass = torch.tensor(
                [
                    -0.0156557281, -0.0727326195, 0.3848648469, 0.8525720202,
                    0.3378976625, -0.0727326195, -0.0156557281
                ],
                dtype=torch.float32,
            ) * self.pass_weight
            highpass = torch.tensor(
                [
                    0.0156557281, -0.0727326195, -0.3378976625, 0.8525720202,
                    -0.3848648469, -0.0727326195, 0.0156557281
                ],
                dtype=torch.float32,
            ) * self.pass_weight

        elif self.filter_type == "meyer":
            x = torch.linspace(-1, 1, self.filter_length) 
            lowpass = torch.cos(0.5 * torch.pi * x) * self.pass_weight
            highpass = torch.sin(0.5 * torch.pi * x) * self.pass_weight

        elif self.filter_type == "morlet":
            sigma = self.sigma
            x = torch.linspace(-3 * sigma, 3 * sigma, self.filter_length)
            lowpass = torch.exp(-0.5 * x**2 / sigma**2) * torch.cos(2 * torch.pi * 0.5 * x) * self.pass_weight
            highpass = torch.exp(-0.5 * x**2 / sigma**2) * torch.sin(2 * torch.pi * 0.5 * x) * self.pass_weight

        elif self.filter_type == "mexican_hat":
            sigma = self.sigma
            x = torch.linspace(-3 * sigma, 3 * sigma, self.filter_length)
            gaussian = torch.exp(-0.5 * x**2 / sigma**2)
            lowpass = (1 - x**2 / sigma**2) * gaussian * self.pass_weight
            highpass = (-x / sigma**2) * gaussian * self.pass_weight

        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

        return lowpass, highpass
