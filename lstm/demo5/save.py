    # # Save X and y to CSV files

    
    # # Reshape X from 3D to 2D for CSV storage
    # X_2d = X.reshape(X.shape[0], -1)
    
    # # Create column names for X
    # X_cols = [f'feature_{i}' for i in range(X_2d.shape[1])]
    
    # # Convert to DataFrames
    # X_df = pd.DataFrame(X_2d, columns=X_cols)
    # y_df = pd.DataFrame(y, columns=['target'])
    
    # # Save to CSV
    # X_df.to_csv('X_data.csv', index=False)
    # y_df.to_csv('y_data.csv', index=False)
    
    # print("Data saved to X_data.csv and y_data.csv")