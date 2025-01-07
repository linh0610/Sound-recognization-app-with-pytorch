import os
import torchaudio
from torch.utils.data import Dataset


vocab = [
    "a", "à", "á", "ả", "ã", "ạ", "ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "â", "ầ", "ấ", "ẩ", "ẫ", "ậ",
    "b", "c", "d", "đ", "e", "è", "é", "ẻ", "ẽ", "ẹ", "ê", "ề", "ế", "ể", "ễ", "ệ",
    "g", "h", "i", "ì", "í", "ỉ", "ĩ", "ị",
    "k", "l", "m", "n", "o", "ò", "ó", "ỏ", "õ", "ọ", "ô", "ồ", "ố", "ổ", "ỗ", "ộ", "ơ", "ờ", "ớ", "ở", "ỡ", "ợ",
    "p", "q", "r", "s", "t", "u", "ù", "ú", "ủ", "ũ", "ụ", "ư", "ừ", "ứ", "ử", "ữ", "ự",
    "v", "x", "y", "ỳ", "ý", "ỷ", "ỹ", "ỵ",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ".", ",", "!", "?", ":", ";", "-", "_", "(", ")", "\"", "'", " ", "<unk>"
]
class CustomDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        waveform, sample_rate, alignments = load_data(path)
        return waveform, alignments  # Adjust this as per your requirements

def load_data(path: str):
    if isinstance(path, bytes):
        path = path.decode('utf-8')

    file_name = os.path.splitext(os.path.basename(path))[0]
    audio_path = os.path.join('G:', 'data', f'{file_name}.wav')
    alignment_path = os.path.join('G:', 'data', f'{file_name}.txt')

    waveform, sample_rate = torchaudio.load(audio_path)
    alignments = load_alignments(alignment_path, StringLookup(vocab))  # Adjust as necessary

    return waveform, sample_rate, alignments


def load_alignments(path: str, string_lookup: StringLookup) -> torch.Tensor:
    """
    Reads a text file and converts non-silence tokens to a tensor of character indices.
    Assumes each line has tokens and 'sil' denotes silence, which should be ignored.

    Parameters:
        path (str): Path to the text file containing alignments.
        string_lookup (StringLookup): Instance of StringLookup for character-to-number conversion.

    Returns:
        torch.Tensor: Tensor of character indices.
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    tokens = []
    for line in lines:
        line = line.strip().split()
        # Skip silence tokens
        if len(line) > 2 and line[2] != 'sil':
            tokens.extend([' ', line[2]])

    # Join tokens into a single string and split into characters
    token_string = ''.join(tokens)
    token_chars = list(token_string)

    # Convert characters to indices using the StringLookup instance
    encoded_tensor = string_lookup.encode(token_chars)

    # Return tensor excluding the initial padding index (if any)
    return encoded_tensor[1:] if encoded_tensor.size(0) > 0 else encoded_tensor

string_lookup = StringLookup(vocabulary=vocab)
class StringLookup:
    def __init__(self, vocabulary, oov_token="<OOV>"):
        # Initialize the vocabulary and out-of-vocabulary token
        self.oov_token = oov_token
        self.vocab = vocabulary
        self.char_to_num = {char: idx for idx, char in enumerate(vocabulary)}
        self.num_to_char = {idx: char for idx, char in enumerate(vocabulary)}

        # Add the OOV token
        self.char_to_num[oov_token] = len(vocabulary)
        self.num_to_char[len(vocabulary)] = oov_token

        self.vocab_size = len(self.char_to_num)

    def encode(self, chars):
        """Convert a string of characters to a tensor of numbers."""
        return torch.tensor([self.char_to_num.get(char, self.char_to_num[self.oov_token]) for char in chars],
                            dtype=torch.long)

    def decode(self, nums):
        """Convert a tensor of numbers back to a string of characters."""
        return ''.join(self.num_to_char.get(num.item(), self.oov_token) for num in nums)

    def get_vocabulary(self):
        """Return the vocabulary."""
        return list(self.char_to_num.keys())

    def vocabulary_size(self):
        """Return the size of the vocabulary."""
        return self.vocab_size