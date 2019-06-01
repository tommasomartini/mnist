import mnist.ml.experiment_scheduler as scheduler
import mnist.paths as paths


def main():
    log_dir = paths.BasePaths.EXP_LOG_DIR
    scheduler.ExperimentScheduler(log_dir).run()


if __name__ == '__main__':
    main()
