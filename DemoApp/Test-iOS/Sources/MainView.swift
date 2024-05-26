import SwiftUI

struct MainView: View {
    @StateObject var viewModel = ViewModel()

    var body: some View {
        VStack {
            Text(textStatus())
                .multilineTextAlignment(.center)
        }
        .padding(25)
        .onAppear {
            Task {
                try await viewModel.test()
            }
        }
    }
    
    private func textStatus() -> String {
        switch viewModel.state {
        case .initial:
            return ""
        case .loadingModel:
            return "Loading ml model..."
        case .loadingExampleTensors:
            return "Loading example input and output tensors from json files..."
        case .runningModel:
            return "Running model..."
        case .validation:
            return "Comparing CoreML and PyTorch output tensors..."
        case let .completed(ok):
            let extraInfo = "\nFor more info please read the logs"
            
            if ok {
                return "Validation: successful\n" +
                       "(CoreML and PyTorch output tensors are equal)" +
                        extraInfo
            }
            return "Validation: failed\n" +
                   "(CoreML and PyTorch output tensors aren't equal)" +
                   extraInfo
        }
    }
}
