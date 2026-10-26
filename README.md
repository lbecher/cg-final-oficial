# T-Rex 3D

>Colaboradores: [Heloisa Alves](https://github.com/Helogizzy), [Luiz Becher](https://github.com/lbecher) e [Matheus Jacinto](https://github.com/MatheusLFJacinto)

## Sobre

Versão 3D do T-Rex Game (o jogo do dinossauro do Google Chrome), proposta no Trabalho 2 de Computação Gráfica – turma 2026. O T-Rex é um cubo, os cactos são esferas de poucas faces e um Sol amarelo atravessa o céu como fonte de luz da cena.

Toda a renderização é feita em software, sem OpenGL: pipeline de câmera de Alvy-Ray Smith (SRU → SRC → projeção ortográfica → janela/porta de visão), preenchimento por scanline com z-buffer e sombreamento constante, Gouraud ou Phong (modelo de iluminação ambiente + difuso + especular).

## Como jogar

- Barra de espaços ou clique na tela: inicia o jogo, faz o cubo saltar e reinicia após uma colisão.
- O painel lateral permite trocar o sombreamento e ligar/desligar a ocultação de faces pela normal.

## Compilação e execução

Requer [Rust](https://rustup.rs/).

```sh
cargo run --release
```

O script `build.sh` instala as dependências e gera os executáveis para Linux e Windows.

## Estrutura

- `src/game.rs` – estado e regras do jogo, montagem da cena a cada quadro.
- `src/object.rs` – malhas poligonais (paralelepípedo e esfera) e aplicação de transformações.
- `src/transform.rs` – matrizes de translação, rotação e escala.
- `src/render.rs` – pipeline de visualização, scanline, z-buffer e sombreamentos.
- `src/app/app.rs` – janela, laço de animação, entrada do usuário e painel lateral.
