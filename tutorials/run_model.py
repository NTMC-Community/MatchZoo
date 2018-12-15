import matchzoo as mz
m = mz.models.NaiveModel()
m.guess_and_fill_missing_params()
m.build()

mz.models.list_available()
