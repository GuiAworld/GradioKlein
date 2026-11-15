# GradioKlein 9B (`stable-diffusion.cpp` + Debug Mode)

Interface Gradio para edição e geração de imagens com **FLUX.2 Klein 9B** usando **`stable-diffusion.cpp` (`sd-server`) pré-compilado** (sem compilação de binários C++/CUDA), com pipeline completo de detecção YOLOv8 (pessoas, rosto, atributos corporais e pose) e **Modo Debug Local**.

## Arquivos do Projeto

- [`app.py`](./app.py): Interface Gradio completa (detecção de até 8 pessoas, enquadramento/padding, preservação facial por elipse rotacionada com feathering, detecção de atributos/pose, scheduler estático/dinâmico, custom sigmas, Multi-LoRA e Modo Debug).
- [`sdcpp_backend.py`](./sdcpp_backend.py): Gerenciador do servidor persistente `sd-server` (`stable-diffusion.cpp`) e implementação do backend de cor sólida no Modo Debug.
- [`install.py`](./install.py): Instalador multi-modo (`--mode prod` / `--mode debug`) que baixa os 4 modelos YOLO, baixa o `sd-server` **pré-compilado** (via imagem oficial `ghcr.io/leejet/stable-diffusion.cpp:master-cuda` / GitHub Releases, sem compilar) e os pesos de menor VRAM do repositório `placetrtr/PornMaster-flux2-klein-9b`.
- [`lora_downloader.py`](./lora_downloader.py): Utilitário opcional para download de LoRAs (HuggingFace / Civitai).
- [`gradioklein.ipynb`](./gradioklein.ipynb): Notebook do Google Colab reescrito para atualizar via `git pull`, instalar e rodar o `sd-server` + interface Gradio.
- [`run_debug.bat`](./run_debug.bat): Atalho Windows para testar a interface localmente em Modo Debug usando o Python local da pasta.

## Modelos Utilizados (Menor Consumo de VRAM - FLUX.2 Klein 9B)

De [`placetrtr/PornMaster-flux2-klein-9b`](https://huggingface.co/placetrtr/PornMaster-flux2-klein-9b/tree/main):
- **Diffusion Model (UNet)**: `unet/turbo/PornMaster_Flux2Klein_v4_Turbo_Q4_K.gguf` (4 steps, CFG 1.0)
- **Text Encoder (LLM)**: `text_encoders/Qwen3-VL-8B-Q4_K_S.gguf`
- **VAE**: `vae/full_encoder_small_decoder.safetensors`

E os 4 modelos YOLO de detecção:
- `yolov8n.pt` (detecção de pessoas)
- `yolov8n-face-lindevs.pt` (rosto e ângulo dos olhos)
- `yoloboop.pt` (atributos corporais)
- `yolov8m-pose.pt` (postura e orientação)

## Uso no Google Colab (Produção)

1. Abra [`gradioklein.ipynb`](./gradioklein.ipynb) no Google Colab com GPU ativada.
2. Execute a **Célula 1**: ela executa `git clone` (na primeira vez) ou `git pull` (nas próximas) deste repositório e chama `python install.py --mode prod`, instalando tudo sem compilar binários.
3. (Opcional) Execute a **Célula 2** se quiser baixar LoRAs compatíveis com FLUX.2 Klein 9B para `/content/LoRAs`.
4. Execute a **Célula 3** para iniciar o `sd-server`, o túnel Cloudflare (opcional) e a UI Gradio.

## Uso Local — Modo Debug (`--debug`)

No Modo Debug local:
- **NÃO** baixa nem instala `stable-diffusion.cpp` nem os modelos GGUF 9B.
- Instala normalmente as dependências Python e todos os **4 modelos YOLO** de detecção.
- Todas as funções de recorte, máscara facial, atributos, pose e blend rodam normalmente, mas ao clicar em **Gerar** o sistema retorna uma **imagem preenchida com uma única cor sólida** (em vez de rodar a difusão 9B).

Para rodar no Windows (usando o Python portátil já configurado em `.python` / `.venv`):

```bat
run_debug.bat
```

Ou manualmente:

```powershell
.\.venv\Scripts\python.exe install.py --mode debug
.\.venv\Scripts\python.exe app.py --debug --host 127.0.0.1 --port 7860
```
