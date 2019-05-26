import mnist.custom_utils.readonly as ro


class Names(ro.ReadOnly):
    ###############################################################################################
    # Network input
    ITERATOR_HANDLE = 'iterator_handle'

    ###############################################################################################
    # Placeholders
    # IMAGES_PLACEHOLDER = 'images_placeholder'

    # Logging graph
    AVG_LOSS_PLACEHOLDER = 'avg_loss_placeholder'
    ACCURACY_PLACEHOLDER = 'accuracy_placeholder'

    ###############################################################################################
    # Convolutional backbone
    CONVOLUTIONAL_BACKBONE_SCOPE = 'convolutional_backbone'

    ###############################################################################################
    # Feature processing
    FEATURE_PROCESSING_SCOPE = 'feature_processing'

    FEATURE_MAP_FLAT = 'feature_map_flat'
    FC1 = 'fc1'
    DROPOUT_FC1 = 'dropout_fc1'
    FC2 = 'fc2'
    DROPOUT_FC2 = 'dropout_fc2'

    LOGITS = 'logits'

    ###############################################################################################
    # Loss
    LOSS_SCOPE = 'loss'
    EVALUATION_LOSS = 'evaluation_loss'

    ###############################################################################################
    # Outputs
    OUTPUT_SCOPE = 'output'
    OUTPUT_COLLECTION = 'output'
    PROBABILITIES = 'probabilities'
    PREDICTION = 'prediction'
    NUM_CORRECT_PREDICTIONS = 'num_correct_predictions'
    BATCH_SIZE = 'batch_size'

    ###############################################################################################
    # Operations
    TRAINING_OPERATION = 'train_op'
    DATASET_INIT_OP = 'dataset_init_op'

    ###############################################################################################
    # Summaries

    # Training
    TRAINING_LOSS_SUMMARY = 'training_loss'
    TRAINING_SUMMARY_COLLECTION = 'training_summaries'

    # Evaluation
    EVALUATION_SUMMARY_COLLECTION = 'evaluation_summaries'

    # Logging graph
    AVG_LOSS_SUMMARY = 'avg_loss'
    ACCURACY_SUMMARY = 'accuracy'