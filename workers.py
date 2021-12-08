def _fit_pipeline(pipe,xtrain,xtest,ytrain,ytest):
    start_time = time.time()
    print('Training started')
    pipe.fit(xtrain, ytrain)
    print(f'train duration {(time.time() - start_time)}')
    print("model score: %.3f" % pipe.score(xtest, ytest))
    with open(str(pipe.named_steps['classifier'])[:15]+'.txt', 'w') as f:
        f.write('Model '+str(pipe.named_steps['classifier']))
        f.write('\n')
        f.write(f'train duration {(time.time() - start_time)}')
        f.write('\n')
        f.write("model score: %.3f" % pipe.score(xtest, ytest))