def load_combined_model(model_path,  start_ablation_adapter, end_ablation_adapter, ablation_encoder=True, ablation_decoder=False):
    # one of them should be True, not both.
    assert ablation_encoder != ablation_decoder
    
    adapter_path =  model_path
    adapter_config = json.load(open(adapter_path + "/adapter_config.json"))
    org_config = AutoConfig.from_pretrained(adapter_config['base_model_name_or_path'])
    
    org_model = AutoModelForSeq2SeqLM.from_pretrained(adapter_config['base_model_name_or_path'], config=org_config)
    adap_model = load_adapter_model(args.model_name_or_path)
    combined_model = copy.deepcopy(adap_model)
    if ablation_encoder:
        # adap + org + adap
        adap_1 = adap_model.encoder.block[0:start_ablation_adapter]
        org_ = org_model.encoder.block[start_ablation_adapter:end_ablation_adapter]
        adap_2 = adap_model.encoder.block[end_ablation_adapter:]
        
        for i in range(start_ablation_adapter):
            combined_model.encoder.block[i] = adap_1[i]
        for i in range(start_ablation_adapter, end_ablation_adapter):
            combined_model.encoder.block[i] = org_[i-start_ablation_adapter]
        for i in range(end_ablation_adapter, len(combined_model.encoder.block)):
            combined_model.encoder.block[i] = adap_2[i-end_ablation_adapter]
        combined_model.decoder = adap_model.decoder
    else:
        adap_1 = adap_model.decoder.block[0:start_ablation_adapter]
        org_ = org_model.decoder.block[start_ablation_adapter:end_ablation_adapter]
        adap_2 = adap_model.decoder.block[end_ablation_adapter:]
        for i in range(start_ablation_adapter):
            combined_model.decoder.block[i] = adap_1[i]
        for i in range(start_ablation_adapter, end_ablation_adapter):
            combined_model.decoder.block[i] = org_[i-start_ablation_adapter]
        for i in range(end_ablation_adapter, len(combined_model.decoder.block)):
            combined_model.decoder.block[i] = adap_2[i-end_ablation_adapter]
        combined_model.encoder = adap_model.encoder

    return combined_model
