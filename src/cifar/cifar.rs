pub struct Cifar {
    path: String
}

impl Cifar {
    pub fn new(path: String) -> Cifar {
        
        Cifar { path: path }
    }
}
