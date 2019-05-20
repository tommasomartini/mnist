import mnist.config as config
import mnist.ml.experiment_scheduler as scheduler


def main():
    log_dir = config.GeneralConfig.LOG_DIR
    scheduler.ExperimentScheduler(log_dir).run()


if __name__ == '__main__':
    main()
