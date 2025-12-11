using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using GraphicsRendererUI.ViewModels;

namespace GraphicsRendererUI
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            DataContext = new MainViewModel();
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }
    }
}
