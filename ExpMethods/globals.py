
class GlobalValues:
    torch_models = [
        "lstm".casefold(), 
        "node".casefold(),
        ]
        
    command_line_args = {
        "--input_dir":dict(type=str,default=""),
        "--model_dir":dict(type=str,default="./"),
        "--output_dir":dict(type=str,default=""),
        "--epochs":dict(type=int,default=50),
        "--batch":dict(type=int,default=32),
        "--t_start":dict(type=int,default=20),
        "--tolerance":dict(type=int,default=100),
        "--horizon":dict(type=int,default=30),
        "--debug":dict(action="store_true"),
        "--n_workers":dict(type=int,default=511)
    }

    trainer_params = {
    #    "strategy" : "ddp" #(use if multiple GPUs are available)
        "accelerator" : "auto",
        "precision" : "16-mixed", #(use mixed precision)
        "devices" : 1,
        "log_every_n_steps" : 1,
    #    "auto_lr_find" : True, #(chooses learning rate automatically (DEPRECATED))
        "deterministic" : True, #(reproducibility)
        "enable_progress_bar" : False,
        "enable_model_summary" : False,
        "enable_checkpointing" : False
        }
        
    node_params = {
        "hidden_dim" : 64,
        "sensitivity" : "adjoint",
        "solver" : "dopri5",
        }
        
    lstm_params = {
        "hidden_dim" : 50,
        "n_layers" : 1,
        }
        
    sim_params = {
        "start" : 20,
        "log_n_steps" : 20,
    }
