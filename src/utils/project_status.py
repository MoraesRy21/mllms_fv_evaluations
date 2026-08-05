import sys
import torch

def gpu_status(only_summary: bool = False) -> bool:
    """
    Verifica a disponibilidade do CUDA e retorna True se estiver disponível, False caso contrário.
    """
    print(f"Versão do Python: {sys.version}")
    print(f"Versão do PyTorch instalada: {torch.__version__}")
    print(f"Caminho do executável: {sys.executable}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA disponível: {cuda_available}")
    if cuda_available:
        print(f"Placa detectada: {torch.cuda.get_device_name(0)}")
        print(f"Versão do CUDA: {torch.version.cuda}")
        if only_summary:
            print(torch.cuda.memory_summary(device=None, abbreviated=False))
        else:
            print(f"Número de GPUs: {torch.cuda.device_count()}")
            print(f"Propriedades da GPU: {torch.cuda.get_device_properties(0)}")
    else:
        print("Ainda não detectou a GPU.")

    return cuda_available