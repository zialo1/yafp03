#uncomment these if you want to print latex code
#import __main__
##    print(__main__.main_2latex(df,"atable"))


def doallwork(df):
    import __main__

    #afmt=__main__.main_askfmt(len(df.columns))
    afmt = dict(zip(df.columns,['{:.0f}','{:.3e}','{:.3e}']))

    print(__main__.main_2latex(df,"atable",afmt))


