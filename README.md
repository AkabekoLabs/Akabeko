# AkabekoLLM

## AkabekoLLMとは？

小さいモデルをベースにした事前学習や、継続事前学習を目的としたプロジェクトです。　

## AkabekoLLMでできること

torchrunを使って分散GPUで、Qwen3ベースのモデルの事前学習や継続事前学習が可能です。


## 各種使い方

Finewebから1Bトークンのデータセットを作成

```
make_data_fineweb_1b.sh
```

2GPUを用いて、Qweb3-0.6Bで継続事前学習

```
./2gpu_continue.sh
```

4GPUを用いて、Qweb3-0.6Bで継続事前学習

```
./4gpu_continue.sh
```

## 今後の拡張

各種学習方法手法の安定化を目指す